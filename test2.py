# import importlib
# import ifera_n as ifera
import torch
# import numpy as np
# import pandas as pd
# import datetime as dt

# importlib.reload(ifera)

# import os

# os.environ['TORCH_LOGS'] = "+dynamo"
# os.environ['TORCHDYNAMO_VERBOSE'] = "1"

# torch.set_float32_matmul_precision('high')

# config = ifera.InstrumentConfig()
# instrument = config.get_config("CL@IBKR:1m")

# device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
# dtype = torch.float32

# dates, main_data = ifera.load_instrument_data_tensor(instrument, dtype=dtype, device=device)

# market_sim = ifera.MarketSimulatorIntraday(instrument = instrument, dates=dates, data=main_data)
# env = ifera.IntradayEnv(market_sim=market_sim, batch_size=(32,), window_size=60, start_time_idx=0, max_units=5)

# actornet_hidden = ifera.ActorNetHidden(128, device=device, dtype=dtype)
# actor_net = ifera.ActorNet(env, 32, actornet_hidden, dist_return='mode')

# rewards = env.rollout_all(actor_net, env.steps + 1)

# rewards
print(torch.cuda.is_available())

x = torch.tensor([1, 2, 3], device="cuda:0")
