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

import time
import torch
from tqdm import tqdm

t1 = torch.arange(0, 5, device='cuda', dtype=torch.float32)
t2 = torch.arange(0, 10, device='cuda', dtype=torch.float32)

# pb1 = tqdm.tqdm(t1, 'Outer loop', position=0)

# for i in pb1:
#     pb2 = tqdm.tqdm(t2, 'Inner loop', position=1)
#     for j in pb2:
#         time.sleep(0.1)

# pb1 = tqdm(total=t1.numel(), position=0, desc='Outer loop')
# pb2 = tqdm(total=t2.numel(), position=1, desc='Inner loop')

# with pb1:
#     for i in t1:
#         pb1.update(1)
#         pb2.reset()
#         with pb2:
#             for j in t2:
#                 pb2.update(1)
#                 time.sleep(0.1)

pb1 = tqdm(total=len(t1), desc='Outer loop', position=0)
pb2 = tqdm(total=len(t2), desc='Inner loop', position=1)

for i in range(len(t1)):
    pb2.reset()

    for j in range(len(t2)):
        time.sleep(0.3)
        pb2.update(1)

    pb1.update(1)

