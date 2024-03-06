import importlib
import ifera
import torch

from torchrl.modules import ProbabilisticActor
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer, SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

from collections import defaultdict
from tqdm import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type

importlib.reload(ifera)

torch.cuda.empty_cache()

device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

config = ifera.InstrumentConfig()
instrument = config.get_config("CL@IBKR:1m")

dates, main_data = ifera.load_instrument_data_tensor(instrument, dtype=torch.float32, device=device)
proc_data = ifera.pre_process_data(main_data)

market_sim = ifera.MarketSimulatorIntraday(instrument = instrument, dates=dates, data=main_data)

start_time_idx = int((instrument.liquidStart - instrument.tradingStart).total_seconds() / instrument.timeStep.total_seconds())
end_time_idx = int((instrument.liquidEnd - instrument.tradingStart).total_seconds() / instrument.timeStep.total_seconds()) - 1
steps = end_time_idx - start_time_idx

env = ifera.IntradayEnv(market_sim=market_sim, batch_size=(32,), window_size=60, start_time_idx=start_time_idx, end_time_idx=end_time_idx)

n_dim = 16

actor_net = ifera.ActorNetHidden(256, n_dim, device=device, dtype=torch.float32)
policy_module = ifera.PolicyModule(env, n_dim, proc_data, actor_net)

value_net = ifera.ActorNetHidden(256, n_dim, device=device, dtype=torch.float32)
value_module = ifera.ValueModule(env, n_dim, proc_data, value_net)

actor = ProbabilisticActor(module=policy_module, in_keys="logits", spec=env.action_spec, distribution_class=torch.distributions.Categorical, return_log_prob=True)

# Init Lazy Modules to get the correct shape for the input tensors
td = env.reset()
_ = actor(td)
_ = value_module(td)
_ = env.rollout(steps, actor)

frames_per_batch = 32 * steps
epochs = 128
total_frames = frames_per_batch * epochs
sub_batch_size = 64
collector = SyncDataCollector(env, actor, frames_per_batch=frames_per_batch, total_frames=total_frames, split_trajs=False, device=device, reset_at_each_iter=True)

replay_buffer = ReplayBuffer(storage=LazyTensorStorage(frames_per_batch, device=device), sampler=SamplerWithoutReplacement())

advantage_module = GAE(gamma=1.0, lmbda=0.95, value_network=value_module, average_gae=True, device=device)

loss_module = ClipPPOLoss(actor=actor, critic=value_module, clip_epsilon=0.2, entropy_bonus=True, entropy_coef=0.01, critic_coef=0.5)

optim = torch.optim.Adam(loss_module.parameters(), 3e-4)
scheduler = torch.optim.lr_scheduler.LinearLR(optim, 1.0, 0.0, total_iters=epochs)

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(10):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        with torch.no_grad():
            advantage_module(tensordict_data)

        data_view = tensordict_data.reshape(-1)

        replay_buffer.extend(data_view)

        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optim step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            # torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item() * steps)
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.6f}"

    if i % 1 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our env horizon).
        # The ``rollout`` method of the env can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
#        with torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(steps, actor)
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item() / env.batch_size.numel()
            )
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()
