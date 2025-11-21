import pandas as pd
import talib
import talib.abstract as ta
import numpy as np
from numba import njit

def calculate_smi(df, k_length=10, d_length=3):
    # 计算周期内最高价和最低价
    df['hh'] = df['high_price'].rolling(k_length).max()
    df['ll'] = df['low_price'].rolling(k_length).min()
    
    # 计算差值
    df['diff'] = df['hh'] - df['ll']
    df['rdiff'] = df['close_price'] - (df['hh'] + df['ll']) / 2

    # 双重EMA计算
    # 第一重EMA
    ema_rdiff1 = talib.EMA(df['rdiff'].values, timeperiod=d_length)
    ema_diff1 = talib.EMA(df['diff'].values, timeperiod=d_length)
    
    # 第二重EMA
    avgrel = talib.EMA(ema_rdiff1, timeperiod=d_length)
    avgdiff = talib.EMA(ema_diff1, timeperiod=d_length)

    # 计算SMI值
    with np.errstate(divide='ignore', invalid='ignore'):
        smi = np.divide(avgrel, (avgdiff / 2)) * 100
    smi = np.nan_to_num(smi, nan=0.0)  # 处理NaN和无穷大

    # 计算信号线
    signal = talib.EMA(smi, timeperiod=d_length)

    # 将结果添加到DataFrame
    df['smi'] = signal

    # 清理中间列
    df.drop(['hh', 'll', 'diff', 'rdiff'], axis=1, inplace=True)
    
    return df['smi']

def calculate_ema8(df,timeperiod=8):
    """
    计算8日EMA
    """
    return talib.EMA(df['close_price'], timeperiod=timeperiod)

def calculate_ema50(df,timeperiod=50):
    """
    计算50日EMA
    """
    return talib.EMA(df['close_price'], timeperiod=timeperiod)

def calculate_ema100(df,timeperiod=100):
    """
    计算100日EMA
    """
    return talib.EMA(df['close_price'], timeperiod=timeperiod)

def calculate_ema200(df,timeperiod=200):
    """
    计算200日EMA
    """
    return talib.EMA(df['close_price'], timeperiod=timeperiod)


def calculate_crsi(df:pd.DataFrame):
    crsi_closechange = df['close_price'] / df['close_price'].shift(1)
    crsi_updown = np.where(crsi_closechange.gt(1), 1.0, np.where(crsi_closechange.lt(1), -1.0, 0.0))
    crsi=(talib.RSI(df['close_price'], timeperiod=3) + talib.RSI(crsi_updown, timeperiod=2) + talib.ROC(df['close_price'], 100)) / 3
    return crsi

# Williams %R
def williams_r(dataframe: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high_price"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low_price"].rolling(center=False, window=period).min()

    WR = pd.Series(
        (highest_high - dataframe['close_price']) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

def calculate_R_96(df):
    r96=williams_r(df, period=96)
    return r96

def calculate_R_480(df):
    r480=williams_r(df, period=480)
    return r480

def range_percent_change(dataframe: pd.DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        if method == 'HL':
            return (dataframe['high_price'].rolling(length).max() - dataframe['low_price'].rolling(length).min()) / dataframe['low_price'].rolling(length).min()
        elif method == 'OC':
            return (dataframe['open_price'].rolling(length).max() - dataframe['close_price'].rolling(length).min()) / dataframe['close_price'].rolling(length).min()
        else:
            raise ValueError(f"Method {method} not defined!")


def calculate_H1_prc_change_5(df,periods=5):
    # Range percent change or Rolling Percentage Change Maximum across interval
    return range_percent_change(df, 'HL', length=periods)

#def calculate_ROC(df,periods=21):
#    return talib.ROC(df['close_price'], timeperiod=periods)

def calculate_RSI(df,periods=14):
    return talib.RSI(df['close'], timeperiod=periods)

# Chaikin Money Flow
def chaikin_money_flow(dataframe:pd.DataFrame, n=20, fillna=False) -> pd.Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close_price'] - dataframe['low_price']) - (dataframe['high_price'] - dataframe['close_price'])) / (dataframe['high_price'] - dataframe['low_price'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cmf, name='cmf')


def calculate_CMF(df,periods=20):
    return chaikin_money_flow(df, n=periods)

def T3(dataframe, length=5):
    """
    T3 Average by HPotter on Tradingview
    https://www.tradingview.com/script/qzoC9H1I-T3-Average/
    """
    df = dataframe.copy()

    df['xe1'] = ta.EMA(df['close_price'], timeperiod=length)
    df['xe2'] = ta.EMA(df['xe1'], timeperiod=length)
    df['xe3'] = ta.EMA(df['xe2'], timeperiod=length)
    df['xe4'] = ta.EMA(df['xe3'], timeperiod=length)
    df['xe5'] = ta.EMA(df['xe4'], timeperiod=length)
    df['xe6'] = ta.EMA(df['xe5'], timeperiod=length)
    b = 0.7
    c1 = -b * b * b
    c2 = 3 * b * b + 3 * b * b * b
    c3 = -6 * b * b - 3 * b - 3 * b * b * b
    c4 = 1 + 3 * b + b * b * b + 3 * b * b
    df['T3Average'] = c1 * df['xe6'] + c2 * df['xe5'] + c3 * df['xe4'] + c4 * df['xe3']

    return df['T3Average']

def calculate_T3(df,periods=5):
    return T3(df, length=periods)

def calculate_EWO(dataframe, ema_length=5, ema2_length=35):
    df = dataframe.copy()
    ema1 = ta.EMA(df['close'], timeperiod=ema_length)
    ema2 = ta.EMA(df['close'], timeperiod=ema2_length)
    emadif = (ema1 - ema2) / df['low'] * 100
    return emadif

def calculate_LOW5(df:pd.DataFrame):
    return df['low_price'].shift().rolling(5).min()

def calculate_Safe_dump_50(df:pd.DataFrame):
    #derived from H1_prc_change_5
    df['low_5'] = df['low_price'].shift().rolling(5).min()
    df['hl_pct_change_5'] = calculate_H1_prc_change_5(df,periods=5)
    sd50=((df['hl_pct_change_5'] < 0.0004) | (df['close_price'] < df['low_5']) | (df['close_price'] > df['open_price'])).astype(int)
    #sd50=(df['hl_pct_change_5'] < 0.001371 ).astype(int)
    # 清理临时列
    df.drop(['low_5', 'hl_pct_change_5'], axis=1, inplace=True)
    return sd50

def calculate_weighted_price(df: pd.DataFrame):
    """计算VWAP(Volume Weighted Average Price)"""
    # 计算典型价格 (高价+低价+收盘价)/3
    df['typical_price'] = (df['high_price'] + df['low_price'] + df['close_price']) / 3
    # 计算典型价格 * 成交量
    df['price_volume'] = df['typical_price'] * df['volume']
    
    # 计算累计值
    cumulative_pv = df['price_volume'].cumsum()
    cumulative_volume = df['volume'].cumsum()
    
    # 计算VWAP
    vwap = cumulative_pv / cumulative_volume
    
    # 清理临时列
    df.drop(['typical_price', 'price_volume'], axis=1, inplace=True)
    
    return vwap

def calculate_adx(df: pd.DataFrame, period: int = 14):
    adx=talib.ADX(df['high_price'], df['low_price'], df['close_price'], timeperiod=period)
    plus_di=talib.PLUS_DI(df['high_price'], df['low_price'], df['close_price'], timeperiod=period)
    minus_di=talib.MINUS_DI(df['high_price'], df['low_price'], df['close_price'], timeperiod=period)
    return adx, plus_di, minus_di

def calculate_atr(df: pd.DataFrame, period: int = 14):
    atr=talib.ATR(df['high_price'], df['low_price'], df['close_price'], timeperiod=period)
    return atr

def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: int = 2):
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    return upper, middle, lower

def calculate_stochf(df: pd.DataFrame, fastk_period: int = 5, fastd_period: int = 3):
    fastk, fastd = talib.STOCHF(df['high_price'], df['low_price'], df['close_price'], fastk_period=fastk_period, fastd_period=fastd_period)
    return fastk, fastd


def calculate_ema(df,timeperiod=8):
    return talib.EMA(df['close_price'], timeperiod=timeperiod)

def calculate_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    macd, signal, _ = talib.MACD(df['close_price'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    return macd, signal

# ---------------------------------------------

def rolling_weighted_mean(series, window=200, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    try:
        return series.ewm(span=window, min_periods=min_periods).mean()
    except Exception as e:  # noqa: F841
        return pd.ewma(series, span=window, min_periods=min_periods)
# ---------------------------------------------
def calculate_hma_50(df, window=50, min_periods=None):
    min_periods = window if min_periods is None else min_periods
    ma = (2 * rolling_weighted_mean(df['close_price'], window / 2, min_periods)) - rolling_weighted_mean(
        df['close_price'], window, min_periods
    )
    return rolling_weighted_mean(ma, np.sqrt(window), min_periods)
# ---------------------------------------------


def calculate_fast_cti(df,length=20):
    return fast_cti(df['close'], length=length)

import math
# 用 numba 加速计算每个窗口的线性回归相关系数 r
@njit
def _fast_linreg_r(close, length):
    n = close.shape[0]
    # 初始化结果为 nan 数组，前 length-1 个元素保持 nan
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = np.nan

    if n < length:
        return result

    # 固定 x 向量：1, 2, ..., length; 预先计算常量
    x_sum = 0.5 * length * (length + 1)         # sum(x)
    # 根据原代码的写法，x²的和可以这样计算：
    x2_sum = x_sum * (2 * length + 1) / 3.0        # sum(x^2)
    divisor = length * x2_sum - x_sum * x_sum

    # 滚动窗口，从起始位置 i 到 i+length-1
    for i in range(n - length + 1):
        sum_y = 0.0
        sum_xy = 0.0
        sum_y2 = 0.0
        for j in range(length):
            y = close[i + j]
            sum_y += y
            # 注意：x[j] = j+1
            sum_xy += (j + 1) * y
            sum_y2 += y * y

        # 计算相关系数的分子与分母
        numerator = length * sum_xy - x_sum * sum_y
        # 计算第二部分时要先求内层括号
        inner = length * sum_y2 - sum_y * sum_y
        if inner <= 0 or divisor <= 0:
            r_value = np.nan
        else:
            r_value = numerator / math.sqrt(divisor * inner)

        # 将结果放在窗口的最后一位，使得结果序列与原始数据对齐
        result[i + length - 1] = r_value

    return result

def fast_cti(close, length=12, offset=0):
    """
    fast_cti: 基于 numba 优化实现的相关趋势指标 (CTI)
    参数：
         close: 一维数组或 pandas.Series 收盘价序列
         length: 计算滚动窗口的长度，默认为 12
         offset: 位移（shift）参数，默认为 0
    返回：
         pandas.Series 类型的 CTI 值序列，属性 name 设置为 "CTI_{length}"
    """
    # 如果输入是 pandas.Series，则保存原始索引，转换为 numpy 数组
    if isinstance(close, pd.Series):
        index = close.index
        close_arr = close.values.astype(np.float64)
    else:
        index = None
        close_arr = np.asarray(close, dtype=np.float64)

    # 检查 length 合法性
    length = int(length) if (length is not None and length > 0) else 12

    # 调用 numba 加速函数计算线性回归相关系数（r值）
    cti_vals = _fast_linreg_r(close_arr, length)

    # 处理 offset 位移：类似 pandas.Series.shift() 的效果，
    # 如果 offset > 0，则前 offset 个值赋为 nan，
    # 如果 offset < 0，则后 |offset| 个值赋为 nan
    if offset != 0:
        shifted = np.empty_like(cti_vals)
        shifted[:] = np.nan
        if offset > 0:
            # 向下移动：前 offset 个值为 nan
            shifted[offset:] = cti_vals[:-offset]
        else:  # offset < 0
            off = abs(offset)
            shifted[:-off] = cti_vals[off:]
        cti_vals = shifted

    # 转换为 pandas.Series，并命名
    cti_series = pd.Series(cti_vals, index=index)
    cti_series.name = f"CTI_{length}"
    # 可额外设置一个分类属性（如使用 pandas 的扩展属性）
    cti_series.category = "momentum"
    return cti_series

def calculate_vwap(df,length=20):
    """
    计算滚动窗口内的VWAP (Volume Weighted Average Price)
    VWAP = 滚动窗口内的(价格*成交量)之和除以成交量之和，价格通常取 (高价+低价+收盘价)/3
    参数:
      df: 包含 'high', 'low', 'close', 'volume' 的DataFrame
      length: 整数，滚动窗口大小（默认为20）
    返回:
      一个pandas.Series，包含计算得到的VWAP序列
    """
    # 计算典型价格
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    # 计算典型价格乘以成交量
    pv = typical_price * df['volume']
    # 计算滚动窗口内的累计 pv 与成交量
    rolling_pv = pv.rolling(window=length).sum()
    rolling_vol = df['volume'].rolling(window=length).sum()
    # 计算VWAP
    vwap = rolling_pv / rolling_vol
    return vwap

def calculate_vwapb(df, window_size=20, num_of_std=1):
    """
    计算VWAP Bands (成交量加权平均价格带)
    参数:
      df: 包含 'high', 'low', 'close', 'volume' 的DataFrame
      window_size: 用于计算VWAP和标准差的滚动窗口大小，默认为20
      num_of_std: 标准差倍数，默认为1
    返回:
      vwap下轨, VWAP, vwap上轨
    """
    df = df.copy()
    # 使用现有的calculate_vwap函数计算滚动VWAP
    df['vwap'] = calculate_vwap(df, length=window_size)
    rolling_std = df['vwap'].rolling(window=window_size).std()
    df['vwap_low'] = df['vwap'] - (rolling_std * num_of_std)
    df['vwap_high'] = df['vwap'] + (rolling_std * num_of_std)
    return df['vwap_low'], df['vwap'], df['vwap_high']

def calculate_IMIN(df, window_size=20):
    return df['low'].rolling(window=window_size).apply(lambda x: np.argmin(x) / window_size, raw=True)

def calculate_KLEN(df: pd.DataFrame):
    return (df['high'] - df['low']) / df['open']

def calculate_MAX(df: pd.DataFrame, window=10):
    return df['high'].rolling(window=window).max() / df['close']

def calculate_MA(df: pd.DataFrame, window=10):
    return df['close'].rolling(window=window).mean() / df['close']

def calculate_QTLU(df: pd.DataFrame, window=10):
    return df['close'].rolling(window=window).quantile(0.8) / df['close']

def calculate_ROC(df: pd.DataFrame, window=10):
    return df['close'].shift(window) / df['close']

def calculate_LOW(df: pd.DataFrame, window=10):
    return df['low'].rolling(window=window).min() / df['close']

def calculate_QTLD(df: pd.DataFrame, window=10):
    return df['close'].rolling(window=window).quantile(0.2) / df['close']

def rolling_linreg(x):
    t = np.arange(len(x))
    if np.all(np.isnan(x)):
        return (np.nan, np.nan, np.nan)
    try:
        slope, intercept = np.polyfit(t, x, 1)
        pred = slope * t + intercept
        ss_res = np.sum((x - pred) ** 2)
        ss_tot = np.sum((x - np.mean(x)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        resid = x[-1] - (slope * (len(x) - 1) + intercept)
    except Exception:
        slope, r2, resid = np.nan, np.nan, np.nan
    return slope, r2, resid

@njit
def rolling_linreg_fast(x):
    # 只能用支持的基础运算
    n = len(x)
    t = np.arange(n)
    t_mean = np.mean(t)
    x_mean = np.mean(x)
    cov = np.sum((t - t_mean) * (x - x_mean))
    var = np.sum((t - t_mean) ** 2)
    if var == 0:
        slope = np.nan
        intercept = np.nan
    else:
        slope = cov / var
        intercept = x_mean - slope * t_mean
    pred = slope * t + intercept
    ss_res = np.sum((x - pred) ** 2)
    ss_tot = np.sum((x - x_mean) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    resid = x[-1] - (slope * (n - 1) + intercept)
    return slope, r2, resid

def calculate_RESI(df: pd.DataFrame, window=10):
    return df['close'].rolling(window=window).apply(lambda x: rolling_linreg_fast(x)[2], raw=True) / df['close']

def calculate_CORD(df: pd.DataFrame, window=10):
    ret = df['close'] / df['close'].shift(1)
    vol_ret = np.log(df['volume'] / df['volume'].shift(1) + 1)
    return ret.rolling(window=window).corr(vol_ret)

def calculate_SUMP(df: pd.DataFrame, window=10):
    """
    计算上涨天数占比 (正差的天数/绝对变化之和)
    """
    def sump(x):
        pos = np.sum(x > 0)
        denom = np.sum(np.abs(x))
        return pos / (denom + 1e-12)
    diffs = df['close'].diff()
    sump_n = diffs.rolling(window=window).apply(sump, raw=True)
    return sump_n