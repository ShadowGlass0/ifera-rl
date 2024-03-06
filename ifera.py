DATA_FOLDER = "data"
KIBOT_USERNAME = "guest"
KIBOT_PASSWORD = "guest"


SECONDS_IN_DAY = 24 * 60 * 60


from typing import Optional
import pandas as pd
import numpy as np
import pathlib as pl
import requests
import gzip
import datetime
import json
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack
from einops.layers.torch import Rearrange

from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.envs import EnvBase
from torchrl.data.replay_buffers import ReplayBuffer, SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import TensorStorage
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec, BinaryDiscreteTensorSpec, DiscreteTensorSpec
from torchrl.modules import ValueOperator

@dataclass
class InstrumentData:
    symbol: str
    currency: str
    type: str
    broker: str
    interval: str
    tradingStart: str
    tradingEnd: str
    liquidStart: str
    liquidEnd: str
    skipStartTime: str
    contractMultiplier: int
    tickSize: float
    margin: float
    commission: float
    minCommission: float
    maxCommissionPct: float
    slippage: float
    minSlippage: float
    referencePrice: float
    timeStep: pd.Timedelta = None
    endTime: pd.Timedelta = None
    totalSteps: int = None

    def __post_init__(self):
        self.tradingStart = pd.to_timedelta(self.tradingStart)
        self.tradingEnd = pd.to_timedelta(self.tradingEnd)
        self.liquidStart = pd.to_timedelta(self.liquidStart)
        self.liquidEnd = pd.to_timedelta(self.liquidEnd)
        self.skipStartTime = pd.to_timedelta(self.skipStartTime)
        self.timeStep = pd.to_timedelta(self.interval)

        self.endTime = self.tradingEnd - self.tradingStart - self.timeStep
        self.totalSteps = int((self.endTime - self.skipStartTime).total_seconds() / self.timeStep.total_seconds()) + 1



"""
InstrumentConfig class

Loads instrument configuration from a json file.

Constructor parameters
----------------------
filename : str
    Path to json file.
""" 
class InstrumentConfig:
    __instance = None

    def __new__(cls, filename="data/instruments.json", reset=False):
        if InstrumentConfig.__instance is None or reset:
            InstrumentConfig.__instance = object.__new__(cls)
            InstrumentConfig.__instance.data = None
            InstrumentConfig.__instance.filename = filename
        return InstrumentConfig.__instance

    def load_data(self):
        if self.data is None:
            with open(self.filename, 'r') as f:
                self.data = json.load(f)
        return self.data

    """
    Get instrument configuration.

    Parameters
    ----------
    instrument : str
        Instrument config symbol, <symbol>@<broker>:<interval>, e.g. "BTCUSD@bitfinex:1m".

    Returns
    -------
    config : dict
        Instrument configuration.
    """
    def get_config(self, instrument):
        data = self.load_data()
        return InstrumentData(**data[instrument])


def make_path(raw: bool, instrument: InstrumentData, remove_file=False):
    """
    Generate path to csv file.

    Parameters
    ----------
    raw : bool
        If True, load raw data, else load processed data.
    instrument_type : str
        Instrument type, e.g. "crypto", "forex", "stock".
    symbol : str
        Instrument symbol, e.g. "BTCUSD", "EURUSD", "AAPL".
    interval : str
        Interval, e.g. "1m", "1h", "1d".

    Returns
    -------
    path : pathlib.Path
        Path to csv file.
    """
    if raw:
        source = "raw"
    else:
        source = "processed"

    path = pl.Path(DATA_FOLDER, source, instrument.type, instrument.interval, instrument.symbol)
    path = path.with_suffix(".csv")

    # Create the folders in the path if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if remove_file:
        path.unlink(missing_ok=True)

    return path



def load_data(raw: bool, instrument: InstrumentData, dtype="float32"):
    """
    Load data from csv files.

    Parameters
    ----------
    raw : bool
        If True, load raw data, else load processed data.
    instrument_type : str
        Instrument type, e.g. "crypto", "forex", "stock".
    symbol : str
        Instrument symbol, e.g. "BTCUSD", "EURUSD", "AAPL".
    interval : str
        Interval, e.g. "1m", "1h", "1d".

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing data.
    """

    path = make_path(raw, instrument)

    # Check if file exists
    if not path.exists():
        if raw:
            download_data(instrument)
        else:
            process_data(instrument)
    elif not raw:
        # Check if the processed file is older than the raw file
        raw_path = make_path(raw=True, instrument=instrument)
        if raw_path.stat().st_mtime > path.stat().st_mtime:
            process_data(instrument)

    columns = ["date", "time", "open", "high", "low", "close", "volume"] if raw \
        else ["date", "time", "trade_date", "offset_time", "open", "high", "low", "close", "volume"]
    parse_dates = [["date", "time"]] if raw else False
    index_col = ["date_time"] if raw else None

    # In the processed files, dates are represented in ordinal format and times are in seconds, both as integers.
    dtype = {"open": dtype, "high": dtype, "low": dtype, "close": dtype, "volume": "int32"} if raw \
        else {"open": dtype, "high": dtype, "low": dtype, "close": dtype, "volume": "int32", "date": "int32", "time": "int32", "trade_date": "int32", "offset_time": "int32"}

    df = pd.read_csv(path, header=None, parse_dates=parse_dates, names=columns, dtype=dtype, index_col=index_col)

    return df


def download_data(instrument: InstrumentData):
    """
    Download data from Kibot web API.

    Parameters
    ----------
    instrument_type : str
        Instrument type, e.g. "crypto", "forex", "stock".
    symbol : str
        Instrument symbol, e.g. "BTCUSD", "EURUSD", "AAPL".
    interval : str
        Interval, e.g. "1m", "1h", "1d".

    Returns
    -------
    None.
    """
    # Raise not implemented error
    raise NotImplementedError("Download data not implemented yet")
    
    # Convert interval to Kibot format: For intraday it's the number of minutes, for daily it's "Daily"
    if interval == "1d":
        interval = "Daily"
    else:
        if interval[-1] == "m":
            interval = int(interval[:-1])
        elif interval[-1] == "h":
            interval = int(interval[:-1]) * 60

    url = f"http://api.kibot.com/?action=history&symbol={symbol}&interval={interval}&type={instrument_type}&splitadjusted=1&user={KIBOT_USERNAME}&password={KIBOT_PASSWORD}"
    
    print(url)

    headers = {'Accept-Encoding': 'gzip'}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        decompressed_content = gzip.decompress(response.content)
        path = make_path(raw=True, instrument_type=instrument_type, interval=interval, symbol=symbol, remove_file=True)

        with open(path, 'w') as f:
            f.write(decompressed_content.decode())
    else:
        print(f"Failed to download data. HTTP Status code: {response.status_code}")


def process_data(instrument: InstrumentData):
    """
    Process raw data and save to csv file.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.

    Returns
    -------
    None.
    """
    
    time_step = instrument.timeStep
    start_time_offset = instrument.tradingStart
    start_time = instrument.skipStartTime
    end_time = instrument.endTime
    total_steps = instrument.totalSteps
    offset_seconds = start_time_offset.total_seconds()

    df = load_data(raw=True, instrument=instrument, dtype="float64")
    
    df['date'] =  pd.to_datetime(df.index.date)
    df['time'] =  pd.to_timedelta(df.index.hour * 3600 + df.index.minute * 60 + df.index.second, unit='s')

    # Set offset_time to the time from the start of the trading day    
    df['offset_time'] = df['time'] - start_time_offset
    # Remove the day component from offset_time
    df['offset_time'] = df['offset_time'].apply(lambda x: x - pd.to_timedelta(x.days, unit='d'))

    # Calculate the trade_date, which is the date of the trading day. Could be the previous day, for example futures trade starting at 6pm on the previous day.
    df['trade_date'] = pd.to_datetime((df.index - start_time_offset).date)

    full_days = load_full_days()

    # Filter out the days that are not full trading days
    df = df[df["trade_date"].isin(full_days.index)]

    # Group the DataFrame by date and apply the function to each group
    df = df.groupby('trade_date').apply(add_missing_rows, start_time=start_time, end_time=end_time, time_step=time_step).reset_index(drop=True)

    # Filter out the rows that are outside the trading hours
    df = df[(df['offset_time'] >= start_time) & (df['offset_time'] <= end_time)]

    # Filter out the days that don't have the expected number of rows
    df = df.groupby('trade_date').filter(lambda x: x['open'].count() == total_steps)

    df.sort_values(['trade_date', 'offset_time'], inplace=True)

    df['ord_trade_date'] = df['trade_date'].map(lambda x: x.toordinal()).astype('int32')
    df['time_seconds'] = (df['offset_time'].map(lambda x: x.total_seconds()) + offset_seconds).mod(SECONDS_IN_DAY).astype('int32')
    df['offset_time_seconds'] = (df['offset_time'].map(lambda x: x.total_seconds())).astype('int32')
    df['ord_date'] = df['ord_trade_date'] + ((df['offset_time_seconds'] + offset_seconds) // SECONDS_IN_DAY).astype('int32')

    df = df[['ord_date', 'time_seconds', 'ord_trade_date', 'offset_time_seconds', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    # Save to csv file
    path = make_path(raw=False, instrument=instrument, remove_file=True)
    df.to_csv(path, header=False)


def generate_full_days(symbol):
    """
    Generate a file with all full trading days.
    A raw stock file with 30 minute intervals is required.

    Parameters
    ----------
    symbol : str
        Instrument symbol, e.g. "BTCUSD", "EURUSD", "AAPL".
    
    Returns
    -------
    None.
    """

    config = InstrumentConfig()
    instrument = config.get_config(f"{symbol}@IBKR:30m")

    # Read the file into a dataframe (date, time, open, high, low, close, volume). This should be a csv file with 30 minute intervals.
    df = load_data(raw=True, instrument=instrument)

    df['Date'] = df.index.date
    df['Time'] = df.index.time

    start_time = datetime.time(9, 30)
    end_time = datetime.time(15, 30)

    # Create a new DataFrame with one line for each date, with date, min_time and max_time
    dates = df.groupby('Date').agg({'Time': ['min', 'max']}).reset_index()

    # Filter out the dates where the first time for the day is larger than 9:30 or the last time for the day is less than 15:30
    short_days = dates[(dates['Time']['min'] > start_time) | (dates['Time']['max'] < end_time)]
    full_days = dates[~dates['Date'].isin(short_days['Date'])]['Date']

    full_days_path = pl.Path(DATA_FOLDER, "processed", "full_days.csv")
    short_days_path = pl.Path(DATA_FOLDER, "processed", "short_days.csv")
    full_days_path.unlink(missing_ok=True)
    short_days_path.unlink(missing_ok=True)
    
    # Write full_days into a file for later use
    full_days.to_csv(full_days_path, index=False, header=False)

    # Write short_days into a file for later use, including the min and max times
    short_days.to_csv(short_days_path, index=False)


def load_full_days():
    #TODO: Remove magic strings
    path = pl.Path(DATA_FOLDER, "processed", "full_days.csv")

    # Check if file exists
    if not path.exists():
        generate_full_days("IBM")

    return pd.read_csv(path, header=None, names=["date"], parse_dates=["date"], index_col="date")


def add_missing_rows(group, start_time, end_time, time_step):
    """
    This function takes a DataFrame `group` and a time range defined by `start_time`, `end_time`, and `time_step`.
    It creates a new DataFrame that includes all time steps in the given range, not just those present in the input DataFrame.
    For each new row (time step) added, it fills in missing data as follows:
    - 'open', 'high', 'low', 'close' columns are filled with the previous 'close' value
    - 'volume' is filled with 0
    The function returns the new DataFrame with added rows and filled missing data.
    """

    all_time_steps = pd.timedelta_range(start=start_time, end=end_time, freq=time_step)

    # Create a DataFrame with the all time steps for the day
    all_time_step_rows = pd.DataFrame({'trade_date': group['trade_date'].iloc[0], 'offset_time': all_time_steps, 'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan, 'volume': np.nan})

    # Merge the all_time_step_rows with the group, so that we have a row for each time step
    merged = pd.merge(group, all_time_step_rows, on=['trade_date', 'offset_time'], how='outer', suffixes=('', '_y'), sort=True)[group.columns]

    # Fill in missing data: open, high, low, close = previous close, volume = 0
    merged['close'] = merged['close'].ffill()
    merged[['open', 'high', 'low', 'close']] = merged[['open', 'high', 'low', 'close']].bfill(axis=1)
    merged['volume'] = merged['volume'].fillna(0).astype('int32')
    

    return merged


def sma(x: torch.Tensor, window: int):
    """
    Calculate the simple moving average of a tensor. The first element is prepended window-1 times for calculating the first window-1 averages.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor. Must be 1-dimensional.
    window : int
        Window size.

    Returns
    -------
    y : torch.Tensor
        Output tensor.
    """
    return torch.cat((torch.full((window-1,), x[0], device=x.device, dtype=x.dtype), x)).unfold(dimension=0, size=window, step=1).mean(dim=1)


def load_instrument_data_tensor(instrument: InstrumentData, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")):
    """
    Load data from csv files and return as a tensor.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    dtype : torch.dtype, optional
        Data type of the tensor. The default is torch.float32.
    device : torch.device, optional
        Device of the tensor. The default is torch.device("cpu").

    Returns
    -------
    data : torch.Tensor
        Tensor containing data.
    """
    df = load_data(raw=False, instrument=instrument, dtype=numpy_dtype_from_torch_dtype(dtype))
    dates = torch.tensor(np.unique(df['trade_date']), dtype=torch.int32, device=device)
    data = torch.tensor(df[['time', 'open', 'high', 'low', 'close', 'volume']].values, dtype=dtype, device=device)
    data = rearrange(data, '(d t) c -> d t c', t=instrument.totalSteps) 
    return dates, data


def pre_process_data(data: torch.Tensor, volume_sma_window: int = 60) -> torch.Tensor:
    """
    Pre-process data for training.

    Parameters
    ----------
    data : torch.Tensor
        Tensor containing data.

    Returns
    -------
    data : torch.Tensor
        Tensor containing data.
    """
    result = torch.empty((data.shape[0], data.shape[1], 9), dtype=data.dtype, device=data.device)
    
    # Column 0,1 : sin(time) and cos(time) stretched to the range of 0 to 2pi from 0 to seconds_per_day
    time = data[:, :, 0]
    time = time / SECONDS_IN_DAY * 2 * math.pi
    result[:, :, 0] = torch.sin(time)
    result[:, :, 1] = torch.cos(time)

    # Calculate prev_close into a separate tensor. Use open from the df for the first time step of each day and close from the previous time step for the rest.
    prev_close = torch.cat((data[:, 0, 1].unsqueeze(-1), data[:, :-1, 4]), dim=1)

    # Column 2,3,4,5,6,7 : log(high/low), log(high/close), log(close/low), log(close/prev_close), log(high/prev_close), log(low/prev_close)
    result[:, :, 2] = (data[:, :, 2] / data[:, :, 3]).log()
    result[:, :, 3] = (data[:, :, 2] / data[:, :, 4]).log()
    result[:, :, 4] = (data[:, :, 4] / data[:, :, 3]).log()
    result[:, :, 5] = (data[:, :, 4] / prev_close).log()
    result[:, :, 6] = (data[:, :, 2] / prev_close).log()
    result[:, :, 7] = (data[:, :, 3] / prev_close).log()

    result[:, :, 2:8] = F.layer_norm(result[:, :, 2:8], result[:, :, 2:8].shape)

    average_daily_volume = data[:, :, 5].mean(dim=1)
    sma_volume = sma(average_daily_volume, volume_sma_window).unsqueeze(1)

    result[:, :, 8] = data[:, :, 5] / sma_volume - 1.0
    result[:, :, 8] = F.layer_norm(result[:, :, 8], result[:, :, 8].shape)

    return result


def numpy_dtype_from_torch_dtype(dtype):
    """
    Convert a torch dtype to a numpy dtype.

    Parameters
    ----------
    dtype : torch.dtype
        Torch dtype.

    Returns
    -------
    dtype : np.dtype
        Numpy dtype.
    """
    return torch.empty((), dtype=dtype).numpy().dtype


class MarketSimulatorIntraday:
    def __init__(self, instrument: InstrumentData, dates: torch.tensor, data: torch.tensor, pos_dtype: torch.dtype = torch.int32):
        self.instrument = instrument
        self.dates = dates
        self.data = data
        self.dtype = data.dtype
        self.pos_dtype = pos_dtype
        self.device = data.device
        self.channels = { "time":0, "open":1, "high":2, "low":3, "close":4, "volume":5 }
        self.use_max_commission_mask = torch.tensor(self.instrument.maxCommissionPct > 0.0, dtype=torch.bool, device=self.device)
        self.slippage_pct = self.instrument.slippage / self.instrument.referencePrice

    def start_day(self, date_idx: torch.Tensor):
        """
        Start a new trading day. date_idx can be a tensor with multiple indices for batch processing.

        Parameters
        ----------
        date_idx : torch.Tensor
            Indices of the dates in the dates tensor.

        Returns
        -------
        None.
        """
        self.date_idx = date_idx
        self.date = self.dates[date_idx]
        self.time_idx = 0
        self.position = torch.zeros((len(date_idx),), dtype=self.pos_dtype, device=self.device)
        self.avg_entry_price = torch.zeros((len(date_idx),), dtype=self.dtype, device=self.device)

    #@torch.compile(fullgraph=True, mode="max-autotune")
    #@torch.compile()
    def calculate_step(self, date_idx, time_idx, position, avg_entry_price, action) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Calculate the result of a trading step given the current state and an action.

        Parameters
        ----------
        date_idx : torch.Tensor
            Indices of the dates in the dates tensor.
        time_idx : torch.Tensor
            Indices of the time steps in the data tensor.
        position : torch.Tensor
            Current position. Positive for long positions, negative for short positions, and zero for no position. dtype must be self.pos_dtype.
        avg_entry_price : torch.Tensor
            Average entry price of the current position.
        action : torch.Tensor
            Action to take. 0 = do nothing, positive = buy, negative = sell. The absolute value is the number of contracts to buy or sell. dtype must be self.pos_dtype.

        Returns
        -------
        profit : torch.Tensor
            Profit from closing positions during this step.
        new_position : torch.Tensor
            New position after the action is taken.
        new_avg_entry_price : torch.Tensor
            New average entry price after the action is taken.

        This function calculates the profit from closing positions, the new position, and the new average entry price after taking the action. The action can extend the current position, shrink the current position, or switch the direction of the current position. The function also takes into account the commission for the action and the slippage in the execution price.
        """
        
        action_sign = torch.sign(action)
        action_abs = torch.abs(action)
        position_sign = torch.sign(position)
        position_abs = torch.abs(position)
        nz_action = action != 0

        extend_mask = ((action_sign == position_sign) | (position_sign == 0)) & nz_action
        shrink_mask = ((action_sign != position_sign) & (action_abs <= position_abs)) & nz_action
        switch_mask = ((action_sign != position_sign) & (action_abs > position_abs)) & (position_sign != 0)
        not_switch_mask = ~switch_mask

        close_mask = shrink_mask | switch_mask
        open_mask = extend_mask | switch_mask

        new_position = position + action
        close_position = (position - new_position * not_switch_mask) * close_mask
        open_position = (new_position - position * not_switch_mask) * open_mask

        current_price = self.data[date_idx, time_idx, self.channels["open"]]
        slippage = (current_price * self.slippage_pct).clamp(min=self.instrument.minSlippage) * action_sign
        execution_price = current_price + slippage

        prev_position = position * not_switch_mask
        new_avg_entry_price = torch.where((prev_position + open_position) == 0, 0.0, (avg_entry_price * prev_position + execution_price * open_position) / (prev_position + open_position))

        commission = (action_abs * self.instrument.commission).clamp(min=self.instrument.minCommission)
        max_commission = execution_price * action_abs * self.instrument.maxCommissionPct
        commission = torch.where(self.use_max_commission_mask, commission.clamp(max=max_commission), commission)
        
        profit = close_position * (execution_price - avg_entry_price) * self.instrument.contractMultiplier - commission

        return profit, new_position, new_avg_entry_price

    def step(self, action: torch.Tensor) -> bool:
        """
        Perform a trading step given an action.

        Parameters
        ----------
        action : torch.Tensor
            Action to take. 0 = do nothing, positive = buy, negative = sell. The absolute value is the number of contracts to buy or sell. dtype must be self.pos_dtype.

        Returns
        -------
        done : bool
        """
        _, self.position, self.avg_entry_price = self.calculate_step(self.date_idx, self.time_idx, self.position, self.avg_entry_price, action)
        self.time_idx += 1

        return self.time_idx >= self.instrument.totalSteps

class IntradayEnv(EnvBase):
    def __init__(
        self,
        market_sim: MarketSimulatorIntraday,
        batch_size: torch.Size,
        date_idx: torch.Tensor = None,
        window_size: int = 20,
        unit_size: int = 1,
        max_units: int = 5,
        seed: int = None,
        start_time_idx: int = None,
        end_time_idx: int = None,
        reward_scaling: float = 1.0,
        passive_penalty: float = 0.0
    ) -> None:
        super().__init__(device=market_sim.data.device, dtype=market_sim.data.dtype, batch_size=batch_size)
        
        self.instrument = market_sim.instrument
        self.window_size = window_size
        self.max_units = max_units
        self.unit_size = unit_size
        self.market_sim = market_sim
        self.dates = market_sim.dates
        self.data = market_sim.data
        self.dtype = market_sim.dtype
        self.pos_dtype = market_sim.pos_dtype
        self.channels = market_sim.channels
        self.n_position = 2 * max_units + 1
        self.reward_scaling = reward_scaling
        self.passive_penalty = passive_penalty
        self.margin_multiplier = self.instrument.margin / (self.instrument.contractMultiplier * self.instrument.referencePrice)
        
        # A tensor with a subset of indices of the dates in the dates tensor.
        self.date_idx = torch.arange(len(self.dates), dtype=torch.int64, device=self.device) if date_idx is None else date_idx

        if seed is not None:
            self.set_seed(seed)

        # Set up a ReplayBuffer for sampling random windows
        storage = TensorStorage(storage=self.date_idx, device=self.device)
        self.replay_buffer = ReplayBuffer(storage=storage, sampler=SamplerWithoutReplacement())
        
        min_data_value = repeat(self.data[:, :, 1:].min(dim=1)[0].min(dim=0)[0], 'c -> w c', w = self.window_size)
        max_data_value = repeat(self.data[:, :, 1:].max(dim=1)[0].max(dim=0)[0], 'c -> w c', w = self.window_size)

        self.position_spec = DiscreteTensorSpec(shape=batch_size, n=self.n_position, dtype=torch.int64, device=self.device)
        self.observation_spec = CompositeSpec(
            date_idx = BoundedTensorSpec(shape=batch_size, dtype=torch.int64, low=0, high=len(self.dates) - 1, device=self.device),
            time_idx = BoundedTensorSpec(shape=batch_size, dtype=torch.int64, low=self.window_size, high=self.instrument.totalSteps - 1, device=self.device),
            date = BoundedTensorSpec(shape=batch_size, dtype=torch.int32, low=self.dates.min(), high=self.dates.max(), device=self.device),
            time = BoundedTensorSpec(shape=batch_size, dtype=torch.float32, low=0, high=SECONDS_IN_DAY - 1, device=self.device),
            position = self.position_spec,
            avg_entry_price = BoundedTensorSpec(shape=batch_size, dtype=self.dtype, low=min_data_value[0, self.channels['low']-1], high=max_data_value[0, self.channels['high']-1], device=self.device),
            trade_count = BoundedTensorSpec(shape=batch_size, dtype=torch.int64, low=0, high=self.instrument.totalSteps, device=self.device),
            shape = batch_size
        )
        self.action_spec = DiscreteTensorSpec(shape=batch_size, n=self.n_position, dtype=torch.int64, device=self.device)
        self.reward_spec = UnboundedContinuousTensorSpec(shape=batch_size + (1, ), dtype=self.dtype, device=self.device)
        self.done_spec = BinaryDiscreteTensorSpec(shape=batch_size + (1,), n=1, dtype=torch.bool, device=self.device)

        self.start_time_idx = self.window_size if start_time_idx is None else max(start_time_idx, self.window_size)
        self.end_time_idx = self.instrument.totalSteps - 1 if end_time_idx is None else min(end_time_idx, self.instrument.totalSteps - 1)

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reset(self, _ : TensorDict) -> TensorDict:
        batch_size = self.batch_size.numel()
        date_idx = self.replay_buffer.sample(batch_size)
        
        while len(date_idx) < batch_size:
            date_idx = torch.cat((date_idx, self.replay_buffer.sample(batch_size - len(date_idx))))
        
        date_idx = date_idx.view(self.batch_size)
        day_open_price = self.data[date_idx, self.start_time_idx, self.channels['open']]
        self.reference_capital = day_open_price * self.instrument.contractMultiplier * self.max_units * self.unit_size * self.margin_multiplier

        out = TensorDict (
            {
                'date_idx': date_idx,
                'time_idx': torch.full(self.batch_size, self.start_time_idx, dtype=torch.int64, device=self.device),
                'date': self.dates[date_idx],
                'time': self.data[date_idx, self.start_time_idx, self.channels['time']],
                'position': torch.full(self.batch_size, self.max_units, dtype=torch.int64, device=self.device),
                'avg_entry_price': torch.zeros(self.batch_size, dtype=self.dtype, device=self.device),
                'trade_count': torch.zeros(self.batch_size, dtype=torch.int64, device=self.device),
                'done': torch.zeros(self.batch_size + (1,), dtype=torch.bool, device=self.device),
            },
            self.batch_size,
            device=self.device
        )

        return out
    
    
    #@torch.compile()
    #@torch.compile(mode="max-autotune")
    def _step(self, tensordict: TensorDict) -> TensorDict:
        date_idx = tensordict['date_idx']
        time_idx = tensordict['time_idx']
        position = tensordict['position'] - self.max_units
        avg_entry_price = tensordict['avg_entry_price']
        target_pos = tensordict['action'] - self.max_units

        new_time_idx = time_idx + 1
        done = new_time_idx >= self.end_time_idx

        # Force to close all positions at the end of the day
        action = torch.where(done, 0, target_pos) - position

        profit, new_position, new_avg_entry_price = self.market_sim.calculate_step(date_idx, time_idx, position, avg_entry_price, action * self.unit_size)
        trade_count = tensordict['trade_count'] + (action != 0).to(torch.int64)

        reward = profit / self.reference_capital * self.reward_scaling

        # Penalize passive trading
        reward -= torch.where(done & (trade_count == 0), self.passive_penalty, 0.0)

        out = TensorDict({
            'date_idx': date_idx,
            'time_idx': new_time_idx,
            'date': self.dates[date_idx],
            'time': self.data[date_idx, new_time_idx, self.channels['time']],
            'position': new_position + self.max_units,
            'avg_entry_price': new_avg_entry_price.clone(),
            'trade_count': trade_count,
            'reward': reward,
            'done': done,
        }, tensordict.shape, device=self.device)

        return out


class DataWindow(nn.Module):
    __constants__ = ['window_size', 'data']
    
    def __init__(self, window_size: torch.int64 , data: torch.Tensor):
        super().__init__()
        self.window_size = window_size
        self.data = data
        self.range = torch.arange(-self.window_size, 0, dtype=torch.int64, device=self.data.device)

    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor) -> torch.Tensor:
        range = self.range.reshape(torch.Size([1 for _ in time_idx.shape]) + self.range.shape)
        
        time_indices = range + time_idx.unsqueeze(-1)
        date_indices = date_idx.unsqueeze(-1)

        return self.data[date_indices, time_indices, :]
    

class EnvOutTransform(nn.Module):
    __constants__ = ['env', 'proc_data', 'out_channels']
    
    def __init__(self, env: IntradayEnv, proc_data: torch.Tensor, out_channels: torch.Tensor):
        super().__init__()
        self.window_size = env.window_size
        self.out_channels = out_channels
        self.position_spec = env.position_spec

        self.data_window = DataWindow(env.window_size, env.data[..., env.channels['high']:env.channels['close'] + 1])
        self.proc_data_window = DataWindow(env.window_size, proc_data)

        device = proc_data.device
        dtype = proc_data.dtype

        self.lin_data = nn.LazyLinear(out_channels, device=device, dtype=dtype)
        self.lin_pos = nn.LazyLinear(out_channels, device=device, dtype=dtype)

    #@torch.compile(mode="max-autotune")
    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor, position: torch.Tensor, avg_entry_price: torch.Tensor) -> torch.Tensor:
        # batch_size, window_size, 3 (high, low, close)
        data = self.data_window(date_idx, time_idx)   
        
        # batch_size, window_size, proc_channels (9)
        proc_data = self.proc_data_window(date_idx, time_idx)
        
        #print(date_idx.shape, time_idx.shape, data.shape, proc_data.shape, position.shape, avg_entry_price.shape)

        # batch_size, env.n_position
        one_hot_pos = self.position_spec.to_one_hot(position, safe=False)
        
        # batch_size
        aep_scaled = torch.where(avg_entry_price > 0.0, (avg_entry_price / data[..., -1, 2]).log(), 0.0)
        
        # batch_size, window_size, 3 (high, low, close)
        #TODO: Pre-compute this & normalize
        data_scaled = (data / rearrange(data[..., -1, 2], '... -> ... () ()')).log()
        
        # batch_size, window_size, proc_channels + 3
        packed_data = torch.cat((data_scaled, proc_data), dim=-1)

        # batch_size, env.n_position + 1
        packed_pos = torch.cat((one_hot_pos, aep_scaled.unsqueeze(-1)), dim=-1)

        # batch_size, window_size, out_channels
        data_out = self.lin_data(packed_data) 

        # batch_size, out_channels
        pos_out = self.lin_pos(packed_pos)

        # batch_size, window_size + 1, out_channels
        out = torch.cat((data_out, pos_out.unsqueeze(-2)), dim=-2)

        return out


class ActorNetHidden(nn.Sequential):
    def __init__(self, hidden_size: int, d_model: int, device, dtype) -> None:
        super().__init__(
            Rearrange('... w c -> ... (w c)'),
            nn.LazyLinear(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.LazyLinear(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.LazyLinear(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.LazyLinear(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            Rearrange('... (w c) -> ... w c', c=d_model)
        )

class MyAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class ActorPostProcess(nn.Sequential):
    def __init__(self, env: IntradayEnv, device, dtype) -> None:
        super().__init__(
            Rearrange('... w c -> ... c w'),
            MyAvgPool(),
            nn.LazyLinear(env.n_position, device=device, dtype=dtype),
        )

    
class ActorNet(nn.Module):
    def __init__(self, env: IntradayEnv, d_model: int, proc_data: torch.Tensor, main_module: nn.Module) -> None:
        super().__init__()
        
        self.end_out_transform = EnvOutTransform(env, proc_data, d_model)
        self.main_module = main_module
        self.post_process = ActorPostProcess(env, device=proc_data.device, dtype=proc_data.dtype)
    
    #@torch.compile(mode="max-autotune")
    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor, position: torch.Tensor, avg_entry_price: torch.Tensor) -> torch.Tensor:
        out = self.end_out_transform(date_idx, time_idx, position, avg_entry_price)
        out = self.main_module(out)
        out = self.post_process(out)
        return out


class ValuePostProcess(nn.Sequential):
    def __init__(self, device, dtype) -> None:
        super().__init__(
            Rearrange('... w c -> ... c w'),
            MyAvgPool(),
            nn.LazyLinear(1, device=device, dtype=dtype),
        )

class ValueNet(nn.Module):
    def __init__(self, env: IntradayEnv, d_model: int, proc_data: torch.Tensor, main_module: nn.Module) -> None:
        super().__init__()
        
        self.end_out_transform = EnvOutTransform(env, proc_data, d_model)
        self.main_module = main_module
        self.post_process = ValuePostProcess(device=proc_data.device, dtype=proc_data.dtype)
    
    #@torch.compile(mode="max-autotune")
    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor, position: torch.Tensor, avg_entry_price: torch.Tensor) -> torch.Tensor:
        out = self.end_out_transform(date_idx, time_idx, position, avg_entry_price)
        out = self.main_module(out)
        out = self.post_process(out)
        return out


class PolicyModule(TensorDictModule):
    def __init__(self, env: IntradayEnv, d_model: int, proc_data: torch.Tensor, main_module: nn.Module) -> None:
        super().__init__(module=ActorNet(env, d_model, proc_data, main_module), 
                         in_keys=['date_idx', 'time_idx', 'position', 'avg_entry_price'],
                         out_keys=['logits'])

class ValueModule(ValueOperator):
    def __init__(self, env: IntradayEnv, d_model: int, proc_data: torch.Tensor, main_module: nn.Module) -> None:
        super().__init__(module=ValueNet(env, d_model, proc_data, main_module), 
                         in_keys=['date_idx', 'time_idx', 'position', 'avg_entry_price'])


if __name__ == "__main__":
    # Optionally get KIBOT_USERNAME and KIBOT_PASSWORD from command line arguments
    import sys
    if len(sys.argv) > 1:
        KIBOT_USERNAME = sys.argv[1]
    if len(sys.argv) > 2:
        KIBOT_PASSWORD = sys.argv[2]

    # generate_full_days("IBM")
    # TODO: Generate only if file doesn't exist or raw file is newer than processed file

    config = InstrumentConfig()
    instrument = config.get_config("CL@IBKR:1m")
    #print(instrument)
    
    # process_data(instrument)
    # df = load_data(raw=False, instrument=instrument)
    # print(df.head(100))

    (dates, data) = load_instrument_data_tensor(instrument, dtype=torch.float32, device=torch.device("cuda"))
    processed_data = pre_process_data(data)
    
    market_sim = MarketSimulatorIntraday(instrument, dates, data)
    market_sim.start_day(torch.tensor([0, 1, 2], dtype=torch.int32, device=torch.device("cuda")))

    action = torch.tensor([0, 1, -1], dtype=torch.int32, device=torch.device("cuda"))

    # Test Open and No actions
    profit, position, avg_entry_price = market_sim.calculate_step(market_sim.date_idx, market_sim.time_idx, market_sim.position, 0.0, action)
    profit, position, avg_entry_price = profit.clone(), position.clone(), avg_entry_price.clone()

    # Test Close & switch
    action = torch.tensor([0, 0, 1], dtype=torch.int32, device=torch.device("cuda"))
    profit, position, avg_entry_price = market_sim.calculate_step(market_sim.date_idx, market_sim.time_idx + 1, position, avg_entry_price, action)

