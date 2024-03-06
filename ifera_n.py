DATA_FOLDER = "data"
KIBOT_USERNAME = "guest"
KIBOT_PASSWORD = "guest"


SECONDS_IN_DAY = 24.0 * 60.0 * 60.0


from typing import Optional, Final
import pandas as pd
import numpy as np
import pathlib as pl
import requests
import gzip
import datetime
import json
import math
import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions import Categorical
import itertools
from einops import rearrange, repeat, pack
from einops.layers.torch import Rearrange


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
    last_update: float = None
    removeDates: list = None

    def __post_init__(self):
        self.tradingStart = pd.to_timedelta(self.tradingStart)
        self.tradingEnd = pd.to_timedelta(self.tradingEnd)
        self.liquidStart = pd.to_timedelta(self.liquidStart)
        self.liquidEnd = pd.to_timedelta(self.liquidEnd)
        self.skipStartTime = pd.to_timedelta(self.skipStartTime)
        self.timeStep = pd.to_timedelta(self.interval)

        self.endTime = self.liquidEnd - self.tradingStart - self.timeStep
        self.totalSteps = int((self.endTime - self.skipStartTime).total_seconds() / self.timeStep.total_seconds()) + 1

        # Convert the removeDates to a list of datetime objects
        if self.removeDates is not None:
            self.removeDates = [pd.to_datetime(date).date() for date in self.removeDates]


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
            self.last_update = pl.Path(self.filename).stat().st_mtime
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
        instrument = InstrumentData(**data[instrument])
        instrument.last_update = self.last_update
        return instrument


def make_path(raw: bool, instrument: InstrumentData, remove_file: bool=False, special_interval: str=None):
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

    interval = special_interval if special_interval is not None else instrument.interval
    path = pl.Path(DATA_FOLDER, source, instrument.type, interval, instrument.symbol)
    path = path.with_suffix(".csv")

    # Create the folders in the path if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)

    if remove_file:
        path.unlink(missing_ok=True)

    return path



def load_data(raw: bool, instrument: InstrumentData, dtype="float32", reset: bool=False):
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

    print(f"Loading {path}")

    # Check if file exists
    if not path.exists() or reset:
        if raw:
            download_data(instrument)
        else:
            process_data(instrument, reset=reset)
    elif not raw:
        # Check if the processed file is older than the raw file
        raw_path = make_path(raw=True, instrument=instrument)
        last_update = path.stat().st_mtime
        if last_update < raw_path.stat().st_mtime or last_update < instrument.last_update:
            process_data(instrument)

    columns = ["date", "time", "open", "high", "low", "close", "volume"] if raw \
        else ["date", "time", "trade_date", "offset_time", "open", "high", "low", "close", "volume"]

    # In the processed files, dates are represented in ordinal format and times are in seconds, both as integers.
    dtype = {"open": dtype, "high": dtype, "low": dtype, "close": dtype, "volume": "int32"} if raw \
        else {"open": dtype, "high": dtype, "low": dtype, "close": dtype, "volume": "int32", "date": "int32", "time": "int32", "trade_date": "int32", "offset_time": "int32"}

    df = pd.read_csv(path, header=None, parse_dates=False, names=columns, dtype=dtype)

    if raw:
        df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.drop(columns=['date', 'time'])
        df = df.set_index('date_time')

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


def process_data(instrument: InstrumentData, reset: bool=False):
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

    full_days = load_full_days(instrument, reset=reset)

    # Remove dates that are in the removeDates list
    if instrument.removeDates is not None:
        full_days2 = full_days.drop(instrument.removeDates, errors='ignore')
        print(f"Removed {len(full_days) - len(full_days2)} rows from {instrument.symbol} due to removeDates")
        full_days = full_days2

    # Filter out the days that are not full trading days
    df = df[df["trade_date"].isin(full_days.index)]

    # Group the DataFrame by date and apply the function to each group
    df = df.groupby('trade_date').apply(add_missing_rows, start_time=start_time, end_time=end_time, time_step=time_step).reset_index(drop=True)

    # Filter out the rows that are outside the trading hours
    df = df[(df['offset_time'] >= start_time) & (df['offset_time'] <= end_time)]

    # Filter out the days that don't have the expected number of rows
    df = df.groupby('trade_date').filter(lambda x: x['open'].count() == total_steps)

    df.sort_values(['trade_date', 'offset_time'], inplace=True)

    df['ord_trade_date'] = (df['trade_date'].map(lambda x: x.toordinal())).astype('int32')
    df['time_seconds'] = (df['offset_time'].map(lambda x: x.total_seconds()) + offset_seconds).mod(SECONDS_IN_DAY).astype('int32')
    df['offset_time_seconds'] = (df['offset_time'].map(lambda x: x.total_seconds())).astype('int32')
    df['ord_date'] = df['ord_trade_date'] + ((df['offset_time_seconds'] + offset_seconds) // SECONDS_IN_DAY).astype('int32')

    df = df[['ord_date', 'time_seconds', 'ord_trade_date', 'offset_time_seconds', 'open', 'high', 'low', 'close', 'volume']].reset_index(drop=True)

    # Save to csv file
    path = make_path(raw=False, instrument=instrument, remove_file=True)
    df.to_csv(path, header=False)


def generate_full_days(instrument: InstrumentData):
    """
    Generate a file with all full trading days.
    A raw stock file with 30 minute intervals is required.

    Parameters
    ----------
    instrument : InstrumentData
        Instrument configuration.
    
    Returns
    -------
    None.
    """

    instrument30min = copy.deepcopy(instrument)
    instrument30min.interval = "30m"
    instrument30min.__post_init__()
    time_steps_seconds = instrument30min.timeStep.total_seconds()

    # Read the file into a dataframe (date, time, open, high, low, close, volume). This should be a csv file with 30 minute intervals.
    df = load_data(raw=True, instrument=instrument30min)

    start_time_offset = instrument30min.tradingStart
    start_time = instrument30min.skipStartTime
    end_time = instrument30min.endTime

    df['time'] = pd.to_timedelta(df.index.hour * 3600 + df.index.minute * 60 + df.index.second, unit='s')
    df['offset_time'] = df['time'] - start_time_offset
    df['offset_time'] = df['offset_time'].apply(lambda x: x - pd.to_timedelta(x.days, unit='d'))

    df['trade_date'] = pd.to_datetime((df.index - start_time_offset).date)
    df = df[(df['offset_time'] >= start_time) & (df['offset_time'] <= end_time)]

    # Create a new DataFrame with one line for each date, with date, min_time and max_time
    dates = df.groupby('trade_date').agg({'offset_time': ['count']})
    max_steps = int((end_time - start_time).total_seconds() / time_steps_seconds) + 1

    # Filter out the dates where the number of steps is less than the expected number of steps
    short_days = dates[(dates['offset_time']['count'] < max_steps)]
    full_days = dates.drop(short_days.index, errors='ignore').reset_index()[['trade_date']]
    short_days = short_days.reset_index()

    full_days_path = make_path(raw=False, instrument=instrument, remove_file=True, special_interval="fulldays")
    short_days_path = make_path(raw=False, instrument=instrument, remove_file=True, special_interval="shortdays")
    
    # Write full_days into a file for later use
    full_days.to_csv(full_days_path, index=False, header=False)

    # Write short_days into a file for later use, including the min and max times
    short_days.to_csv(short_days_path, index=False)


def load_full_days(instrument: InstrumentData, reset: bool=False):
    #TODO: Remove magic strings
    path = make_path(raw=False, instrument=instrument, special_interval="fulldays")

    # Check if file exists
    if not path.exists() or reset:
        generate_full_days(instrument)
    else:
        # Check if the processed file is older than the raw file
        raw_path = make_path(raw=True, instrument=instrument)
        last_update = path.stat().st_mtime
        if last_update < raw_path.stat().st_mtime or last_update < instrument.last_update:
            generate_full_days(instrument)

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
    data = torch.tensor(df[['offset_time', 'open', 'high', 'low', 'close', 'volume']].values, dtype=dtype, device=device)
    data = rearrange(data, '(d t) c -> d t c', t=instrument.totalSteps) 
    return dates, data


class PriceDataWindow(nn.Module):
    def __init__(self, src_data: torch.Tensor, window_size: int = 60, volume_sma_window: int = 60, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.dtype: Final = src_data.dtype if dtype is None else dtype
        self.device: Final = src_data.device
        self.window_size: Final = window_size
        self.volume_sma_window: Final = volume_sma_window
       
        self.register_buffer("price_data", None)
        self.register_buffer("lr_mean", None)
        self.register_buffer("lr_std", None)
        self.register_buffer("vol_mean", None)
        self.register_buffer("vol_std", None)

        col_log_ratios, col_volume = self._pre_process_data(src_data)
        self._normalize_data(col_log_ratios=col_log_ratios, col_volume=col_volume)
        self.price_data = self.price_data.to(dtype=self.dtype)

        assert ~((torch.isinf(self.price_data) | torch.isnan(self.price_data)).any()), "Data contains NaN or infinite values"

        self.range: Final = torch.arange(-self.window_size, 0, dtype=torch.int64, device=self.device)

    def __len__(self) -> int:
        return self.price_data.shape[0]
    
    @property
    def n_channels(self) -> int:
        return self.price_data.shape[-1]

    def forward(self, date_idx: torch.Tensor, time_idx: int) -> torch.Tensor:
        time_indices = self.range.unsqueeze(0) + time_idx
        date_indices = date_idx.unsqueeze(-1)

        return self.price_data[date_indices, time_indices, :]

    def _pre_process_data(self, data) -> None:
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
        price_data = []
        
        # Normalize time to [0, 1]
        time = data[:, :, 0]
        price_data.append(time / SECONDS_IN_DAY)
        # price_data.append(torch.sin(time))
        # price_data.append(torch.cos(time))

        # Calculate prev_close into a separate tensor. Use open from the df for the first time step of each day and close from the previous time step for the rest.
        prev_close = torch.cat((data[:, 0, 1].unsqueeze(-1), data[:, :-1, 4]), dim=1)

        # log(high/low), log(high/close), log(close/low), log(close/prev_close), log(high/prev_close), log(low/prev_close)
        col_log_start = len(price_data)
        price_data.append((data[:, :, 2] / data[:, :, 3]).log())
        price_data.append((data[:, :, 2] / data[:, :, 4]).log())
        price_data.append((data[:, :, 4] / data[:, :, 3]).log())
        price_data.append((data[:, :, 4] / prev_close).log())
        price_data.append((data[:, :, 2] / prev_close).log())
        price_data.append((data[:, :, 3] / prev_close).log())
        col_log_ratios = slice(col_log_start, len(price_data))

        average_daily_volume = data[:, :, 5].mean(dim=1)
        sma_volume = sma(average_daily_volume, self.volume_sma_window).unsqueeze(1)

        price_data.append(data[:, :, 5] / sma_volume - 1.0)
        col_volume = len(price_data) - 1
        
        self.price_data = torch.stack(price_data, dim=-1)

        return col_log_ratios, col_volume
    
    def _normalize_data(self, col_log_ratios: slice, col_volume: int):
        """
        Normalize the data by standardizing the log ratios and volumes.

        This method calculates the mean and standard deviation of the log ratios and volumes
        along the second dimension of the data tensor. It then standardizes the log ratios and volumes
        by subtracting the mean and dividing by the standard deviation.

        Note:
        - The log ratios are located in the third to eighth columns of the data tensor.
        - The volumes are located in the ninth column of the data tensor.

        After normalization, the standardized log ratios and volumes are stored back in the data tensor.

        """
        log_ratios = self.price_data[:, :, col_log_ratios]

        self.lr_mean = log_ratios.mean(dim=1, keepdim=True)
        self.lr_std = log_ratios.std(dim=1, keepdim=True)

        self.price_data[:, :, col_log_ratios] = (log_ratios - self.lr_mean) / self.lr_std

        self.vol_mean = self.price_data[:, :, col_volume].mean(dim=1, keepdim=True)
        self.vol_std = self.price_data[:, :, col_volume].std(dim=1, keepdim=True)

        self.price_data[:, :, col_volume] = (self.price_data[:, :, col_volume] - self.vol_mean) / self.vol_std


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
    def __init__(self, instrument: InstrumentData, data: torch.tensor, dates: torch.tensor):
        self.instrument: Final = instrument
        self.data: Final = data
        self.dates: Final = dates
        self.channels: Final = { "time":0, "open":1, "high":2, "low":3, "close":4, "volume":5 }
        self.use_max_commission_mask: Final = torch.tensor(self.instrument.maxCommissionPct > 0.0, dtype=torch.bool, device=data.device)
        self.slippage_pct: Final = self.instrument.slippage / self.instrument.referencePrice

    #@torch.compile(fullgraph=True, mode="max-autotune")
    #@torch.compile()
    def calculate_step(self, date_idx, time_idx, position, action) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the result of a trading step given the current state and an action.

        Parameters
        ----------
        date_idx : torch.Tensor
            Indices of the dates in the dates tensor.
        time_idx : torch.Tensor
            Indices of the time steps in the data tensor.
        position : torch.Tensor
            Current position. Positive for long positions, negative for short positions, and zero for no position. 
        action : torch.Tensor
            Action to take. 0 = do nothing, positive = buy, negative = sell. The absolute value is the number of contracts to buy or sell. 

        Returns
        -------
        profit : torch.Tensor
            Profit from closing positions during this step.
        new_position : torch.Tensor
            New position after the action is taken.

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
        kept_position = (position - open_position) * not_switch_mask

        current_price = self.data[date_idx, time_idx, self.channels["open"]]
        slippage = (current_price * self.slippage_pct).clamp(min=self.instrument.minSlippage) * action_sign
        execution_price = current_price + slippage

        commission = (action_abs * self.instrument.commission).clamp(min=self.instrument.minCommission)
        max_commission = execution_price * action_abs * self.instrument.maxCommissionPct
        commission = torch.where(self.use_max_commission_mask, commission.clamp(max=max_commission), commission)

        close_price = self.data[date_idx, time_idx, self.channels["close"]]
        prev_close_price = self.data[date_idx, time_idx - 1, self.channels["close"]]

        position_value_delta = (execution_price - prev_close_price) * close_position + \
            (close_price - execution_price) * open_position + \
            (close_price - prev_close_price) * kept_position 
        position_value_delta = position_value_delta * self.instrument.contractMultiplier
        
        profit = position_value_delta - commission
        
        return profit, new_position

class IntradayEnv:
    def __init__(
        self,
        market_sim: MarketSimulatorIntraday,
        batch_size: torch.Size,
        date_idx: torch.Tensor = None,
        window_size: int = 20,
        unit_size: torch.int8 = 1,
        max_units: torch.int8 = 5,
        seed: int = None,
        start_time_idx: int = None,
        end_time_idx: int = None,
        reward_scaling: float = 1.0
    ) -> None:
        self.instrument: Final = market_sim.instrument
        self.window_size: Final = window_size
        self.batch_size: Final = torch.Size(batch_size)
        self.max_units: Final = max_units
        self.unit_size: Final = unit_size
        self.market_sim: Final = market_sim
        self.dates: Final = market_sim.dates
        self.data: Final = market_sim.data
        self.dtype: Final = market_sim.data.dtype
        self.device: Final = market_sim.data.device
        self.channels: Final = market_sim.channels
        self.n_position: Final = 2 * max_units + 1
        self.reward_scaling: Final = reward_scaling
        self.margin_multiplier: Final = self.instrument.margin / (self.instrument.contractMultiplier * self.instrument.referencePrice)
        
        if seed is not None:
            self._set_seed(seed)

        instrument = self.instrument

        # A tensor with a subset of indices of the dates in the dates tensor.
        self.data_indices = torch.arange(len(self.dates), dtype=torch.int64, device=self.device) if date_idx is None else date_idx

        assert self.batch_size.numel() <= self.data_indices.numel(), "Batch size must be smaller than the number of dates"

        ds = TensorDataset(self.data_indices)
        self.loader: Final = DataLoader(ds, batch_size=self.batch_size.numel(), shuffle=True, drop_last=True)
        self.iter_loader: Final = itertools.cycle(self.loader)

        st_idx = int((instrument.liquidStart - instrument.tradingStart - instrument.skipStartTime).total_seconds() / instrument.timeStep.total_seconds())
        et_idx = int((instrument.liquidEnd - instrument.tradingStart - instrument.skipStartTime).total_seconds() / instrument.timeStep.total_seconds()) - 1
        self.start_time_idx: Final = st_idx if start_time_idx is None else max(st_idx, start_time_idx)
        self.end_time_idx: Final = et_idx if end_time_idx is None else min(et_idx, end_time_idx)
        self.steps: Final = self.end_time_idx - self.start_time_idx + 1

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def reset(self, risk_adjustment_factor = None, date_idx: torch.Tensor = None) -> dict:
        self.date_idx = next(self.iter_loader)[0].view(self.batch_size) if date_idx is None else date_idx
        self.time_idx = self.start_time_idx
        self.position = torch.zeros_like(self.date_idx, dtype=torch.int64, device=self.device)
        self.done = torch.zeros_like(self.date_idx, dtype=torch.bool, device=self.device)
        
        day_open_price = self.data[self.date_idx, self.start_time_idx, self.channels['open']]
        risk_adjustment_factor = self.margin_multiplier if risk_adjustment_factor is None else torch.max(risk_adjustment_factor, self.margin_multiplier)
        self.reference_capital = day_open_price * self.instrument.contractMultiplier * self.max_units * self.unit_size * risk_adjustment_factor

        out = {
            'date_idx': self.date_idx,
            'time_idx': self.time_idx,  
            'position': self.position,
            'done': self.done,
            'reward': torch.zeros_like(self.date_idx, dtype=self.dtype, device=self.device),
        }

        return out
    
    def step(self, target_pos: torch.Tensor) -> dict:
        self.time_idx += 1
        self.done = self.done | torch.full_like(self.done, self.time_idx >= self.end_time_idx, device=self.device, dtype=torch.bool)

        # Force to close all positions at the end of the day
        action = torch.where(self.done, 0, target_pos) - self.position

        profit, self.position = self.market_sim.calculate_step(self.date_idx, self.time_idx, self.position, action * self.unit_size)

        reward = profit / self.reference_capital * self.reward_scaling

        out = {
            'date_idx': self.date_idx,
            'time_idx': self.time_idx,  
            'position': self.position,
            'done': self.done,
            'reward': reward,
        }

        return out
    
    def rollout(self, policy: nn.Module, n_steps: int, risk_adjustment_factor: torch.Tensor = None, date_idx: torch.Tensor = None) -> torch.Tensor:
        out = self.reset(risk_adjustment_factor, date_idx)
        cum_reward = out['reward']

        for _ in range(n_steps):
            action = policy(out['date_idx'], out['time_idx'], out['position'])
            out = self.step(action)
            cum_reward += out['reward']

            if out['done'].all():
                break
        
        return cum_reward

    def rollout_all(self, policy: nn.Module, n_steps: int, risk_adjustment_factor: torch.Tensor = None) -> torch.Tensor:
        loader = DataLoader(TensorDataset(self.data_indices), batch_size=self.batch_size.numel(), shuffle=False, drop_last=False)
        cum_reward = torch.zeros_like(self.data_indices, dtype=self.dtype, device=self.device)

        for date_idx in loader:
            cum_reward[date_idx[0]] = self.rollout(policy, n_steps, risk_adjustment_factor, date_idx[0])

        return cum_reward


class EnvOutTransform(nn.Module):
    def __init__(self, env: IntradayEnv, out_channels: torch.Tensor, dtype: torch.dtype = None) -> None:
        super().__init__()
        self.window_size = env.window_size
        self.out_channels = out_channels
        self.max_units = env.max_units
        self.n_position = env.n_position
        self.device = env.device
        self.dtype = dtype if dtype is not None else env.dtype

        self.price_data_window = PriceDataWindow(env.data, window_size=env.window_size, dtype=self.dtype)

        self.lin_data = nn.Linear(self.price_data_window.n_channels, out_channels, device=self.device, dtype=self.dtype)
        self.lin_pos = nn.Linear(self.n_position, out_channels, device=self.device, dtype=self.dtype)

        self.pos = nn.Parameter(torch.zeros((env.window_size + 1, out_channels), dtype=self.dtype, device=self.device))

    #@torch.compile(mode="max-autotune")
    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        # batch_size, window_size, price_data channels
        price_data = self.price_data_window(date_idx, time_idx)
        
        # batch_size, n_position
        one_hot_pos = F.one_hot(position + self.max_units, num_classes=self.n_position).to(dtype=self.dtype)
        
        # batch_size, window_size, out_channels
        data_out = self.lin_data(price_data) 

        # batch_size, out_channels
        pos_out = self.lin_pos(one_hot_pos)

        # batch_size, window_size + 1, out_channels
        out = torch.cat((data_out, pos_out.unsqueeze(-2)), dim=-2)

        return out + self.pos.unsqueeze(0)


class ActorNetHidden(nn.Sequential):
    def __init__(self, hidden_size: int, device, dtype) -> None:
        super().__init__(
            Rearrange('... w c -> ... (w c)'),
            nn.LazyLinear(hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, device=device, dtype=dtype),
            nn.GELU(),
        )

class MyAvgPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class ActorPostProcess(nn.Module):
    def __init__(self, env: IntradayEnv, device, dtype, dist_return: str = 'sample') -> None:
        super().__init__()
        # Rearrange('... w c -> ... c w'),
        # MyAvgPool(),
        self.linear = nn.LazyLinear(env.n_position, device=device, dtype=dtype)
        self.dist_return = dist_return
        self.max_units = env.max_units

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.softmax(self.linear(x), dim=-1)
        dist = Categorical(probs=x)
        
        if self.dist_return == 'sample':
            res = dist.sample()
        elif self.dist_return == 'mode':
            res = dist.mode
        else:
            raise ValueError(f"Invalid dist_return: {self.dist_return}")
        
        return res - self.max_units

    
class ActorNet(nn.Module):
    def __init__(self, env: IntradayEnv, d_model: int, main_module: nn.Module, dist_return: str = 'sample') -> None:
        super().__init__()
        
        self.end_out_transform = EnvOutTransform(env, d_model)
        self.main_module = main_module
        self.post_process = ActorPostProcess(env, device=env.device, dtype=env.dtype, dist_return=dist_return)
    
    #@torch.compile()
    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        out = self.end_out_transform(date_idx, time_idx, position)
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
    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        out = self.end_out_transform(date_idx, time_idx, position)
        out = self.main_module(out)
        out = self.post_process(out)
        return out


class ValueManager:
    def __init__(self, env: IntradayEnv, cache_profits: bool = False, risk_factor_mult: float = 1.5) -> None:
        self.env: Final = env
        self.instrument: Final = env.instrument
        self.cache_profits: Final = cache_profits

        steps = env.steps
        n_dates = len(env.data_indices)
        market_sim = env.market_sim
        data = market_sim.data
        max_units = env.max_units
        unit_size = env.unit_size
        n_position = env.n_position

        self.optimal_values = torch.full((n_dates, steps, n_position), -np.inf, dtype=env.dtype, device=env.device)
        self.worst_values = torch.full((n_dates, steps, n_position), np.inf, dtype=env.dtype, device=env.device)

        date_idx = env.data_indices
        pos = torch.arange(-max_units, max_units + 1, device=env.device, dtype=torch.int32)
        self.target_pos_range = torch.arange(-max_units, max_units + 1, device=env.device, dtype=torch.int32)

        self.date_idx = repeat(date_idx, 'd -> d p1 p2', p1=n_position, p2=n_position)
        self.pos = repeat(pos, 'p1 -> d p1 p2', d=n_dates, p2=n_position)
        self.target_pos = repeat(self.target_pos_range, 'p2 -> d p1 p2', d=n_dates, p1=n_position)
        self.action = self.target_pos_range - pos

        day_open_price = data[date_idx, env.start_time_idx, env.channels['open']]
        reference_capital = day_open_price * instrument.contractMultiplier * max_units * unit_size
        self.reference_capital = repeat(reference_capital, 'd -> d p p2', p=n_position, p2=n_position)

        if cache_profits:
            self.profits = torch.zeros((n_dates, steps, n_position, n_position) , dtype=env.dtype, device=env.device)
            
            # Pre-calculate profits for all possible actions
            for i in range(steps):
                profit, _ = market_sim.calculate_step(date_idx, i + env.start_time_idx, pos, action)
                profit = (profit / reference_capital)
                self.profits[:, i, :, :] = profit

        # Last time step, the action is fixed to 0 target_position (close all positions)
        profit = self._get_profits(-1)
        profit = repeat(profit[:, :, max_units], 'd p -> d p p2', p=n_position, p2=n_position)
        self.optimal_values[:, -1, :] = (profit).max(dim=2)[0]
        self.worst_values[:, -1, :] = (profit).min(dim=2)[0]

        target_pos_idx = self.target_pos + max_units

        for i in range(steps - 2, -1, -1):
            profit = self._get_profits(i)
            self.optimal_values[:, i, :] = (profit + self.optimal_values[date_idx, i + 1, target_pos_idx]).max(dim=2)[0]
            self.worst_values[:, i, :] = (profit + self.worst_values[date_idx, i + 1, target_pos_idx]).min(dim=2)[0]

        self.risk_adjustment_factor: Final = torch.max(-self.worst_values.min(), self.optimal_values.max(), env.margin_multiplier) * risk_factor_mult
        self.optimal_values: Final = self.optimal_values / self.risk_adjustment_factor
        self.reference_capital: Final = self.reference_capital * self.risk_adjustment_factor
        
        if cache_profits:
            self.profits: Final = self.profits / self.risk_adjustment_factor
        
        # We don't need the worst values anymore
        self.worst_values = None

    def _get_profits(self, time_idx: int) -> torch.Tensor:
        if self.cache_profits:
            return self.profits[:, time_idx, :, :]
        
        profit, _ = self.env.market_sim.calculate_step(self.date_idx, time_idx + self.env.start_time_idx, self.pos, self.action)
        return profit / self.reference_capital
    
    def _get_profits_batch(self, date_idx: torch.Tensor, time_idx: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        if self.cache_profits:
            return self.profits[date_idx, time_idx, pos_idx, :]
        
        batch_size = date_idx.numel()
        position = pos_idx - self.env.max_units
        action = repeat(self.target_pos_range, 'p -> b p', b=batch_size) - position
        profit, _ = self.env.market_sim.calculate_step(date_idx, time_idx + self.env.start_time_idx, position, action)
        
        return profit / self.reference_capital[date_idx, pos_idx, action + self.env.max_units].unsqueeze(-1)

    def optimal_action_values(self, date_idx: torch.Tensor, time_idx: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        profit = self._get_profits_batch(date_idx, time_idx, pos_idx)
        
        return profit + self.optimal_values[date_idx, time_idx + 1, :]


class OptimalValueLoss(nn.Module):
    def __init__(self, value_manager: ValueManager, reduction: str = 'mean') -> None:
        super().__init__()
        self.value_manager: Final = value_manager
        self.reduction: Final = reduction

    def forward(self, date_idx: torch.Tensor, time_idx: torch.Tensor, pos_idx: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the optimal value function.

        Parameters
        ----------
        date_idx : torch.Tensor
            Indices of the dates in the dates tensor. Shape: (batch_size,).
        time_idx : torch.Tensor
            Indices of the time steps in the data tensor. Shape: (batch_size,).
        pos_idx : torch.Tensor
            Indices of the positions. Shape: (batch_size,).
        inputs : torch.Tensor
            Input tensor. Shape: (batch_size, n_position). Values are probabilities of taking each action.

        Returns
        -------
        loss : torch.Tensor
            Loss tensor. Shape: (batch_size,).
        """
        optimal_values = self.value_manager.optimal_action_values(date_idx, time_idx, pos_idx)
        value_loss = optimal_values.max(dim=1)[0] - optimal_values
        weighted_squared_loss = (inputs * value_loss).pow(2).sum(dim=1)

        if self.reduction == 'mean':
            return weighted_squared_loss.mean()
        elif self.reduction == 'sum':
            return weighted_squared_loss.sum()
        else:
            return weighted_squared_loss
        


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
    
    market_sim = MarketSimulatorIntraday(instrument, data, dates)

    action = torch.tensor([0, 1, -1], dtype=torch.int32, device=torch.device("cuda"))

    date_idx = torch.tensor([0, 1, 2], dtype=torch.int32, device=torch.device("cuda"))
    time_idx = 0
    position = torch.tensor([0, 0, 0], dtype=torch.int8, device=torch.device("cuda"))

    # Test Open and No actions
    profit, position = market_sim.calculate_step(date_idx, time_idx, position, action)
    profit, position = profit.clone(), position.clone()

    time_idx = 1

    # Test Close & switch
    action = torch.tensor([0, 0, 1], dtype=torch.int32, device=torch.device("cuda"))
    profit, position = market_sim.calculate_step(date_idx, time_idx, position, action)

