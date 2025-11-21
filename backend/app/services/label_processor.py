from calculate_indicator import calculate_atr,calculate_RSI,calculate_fast_cti,calculate_EWO,calculate_bollinger_bands,calculate_vwapb
from calculate_indicator import calculate_IMIN,calculate_KLEN,calculate_MAX,calculate_MA,calculate_QTLU
from calculate_indicator import calculate_ROC,calculate_LOW,calculate_QTLD,calculate_RESI,calculate_SUMP

import tqdm
import numpy as np
import pandas as pd
import talib

RSI_CA_THRESHOLD=25

def resample_1min_to_nmin(df:pd.DataFrame,n=10,offset=None):
    """将1分钟K线数据合成为10分钟K线"""
    # 重置索引
    #df = df.reset_index()
    # 确保datetime列为datetime类型
    df['datetime'] = pd.to_datetime(df['datetime'])
    # 设置datetime为索引
    df = df.set_index('datetime')
    # 定义聚合规则
    agg_dict = {
        'symbol': 'first', 
        'exchange': 'first',
        'interval': 'first',
        'volume': 'sum',
        'turnover': 'sum',
        'open_interest': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }
    # 按10分钟重采样并聚合
    resampled = df.resample(f'{n}min',offset=offset).agg(agg_dict)
    # 重置索引，使datetime重新成为列
    resampled = resampled.reset_index()
    return resampled

def label_sup_order_simple_1_bin_10m_filter(df, window=10,look_forward=10):
    """
    计算每根K线往后window根K线（包括当前），每根K线各自10分钟后涨跌标签（1:上涨，0:下跌或不涨），
    当前Label为这个窗口内1的比例。当前RSI_cumulative_area>Threshold时才赋值Label.
    Args:
        df: 输入DataFrame，要求有'close'价格等
        window: 统计涨跌比例的窗口
    Returns:
        pd.Series，窗口内涨的标签1的占比，否则为np.nan
    """

    df_copy = df.copy()
    df_copy['RSI'] = calculate_RSI(df, 14)
    Threshold = 30

    df_copy['Label'] = np.nan

    # 首先缓存全体的10分钟后涨跌标签
    rise_tag = np.full(len(df_copy), np.nan)
    for i in range(len(df_copy)):
        if i + look_forward < len(df_copy):
            close_now = df_copy['close'].iloc[i]
            close_future = df_copy['close'].iloc[i + look_forward]
            rise_tag[i] = 1 if (close_future - close_now) > 0 else 0
        else:
            rise_tag[i] = np.nan  # 最后不够look_forward的就nan

    # 然后滑动window为每根K线统计比例
    for i in range(len(df_copy)):
        cur_val = df_copy['RSI'].iloc[i]
        if cur_val < Threshold:
            start = max(0, i )
            end = min(len(df_copy),i  + window )  # Python左闭右开，包含当前
            window_rise_tag = rise_tag[start:end]
            valid_count = np.sum(~np.isnan(window_rise_tag))
            if valid_count == 0:
                df_copy.at[i, 'Label'] = np.nan
            else:
                ratio = np.nansum(window_rise_tag == 1) / valid_count
                df_copy.at[i, 'Label'] = ratio
        else:
            df_copy.at[i, 'Label'] = np.nan

    return df_copy['Label']
    



def calculate_label_with_filter(df, window=29, look_forward=10, label_type='up', filter_type='rsi', threshold=None):
    """
    统一的标签计算函数，支持上涨/下跌标签和RSI/CTI过滤
    Args:
        df: 输入DataFrame
        window: 窗口大小（应为奇数）
        look_forward: 预测周期
        label_type: 'up' 或 'down'
        filter_type: 'rsi' 或 'cti'
        threshold: 过滤阈值，如果为None则使用默认值
    Returns:
        pd.Series，标签
    """
    if window % 2 == 0:
        raise ValueError("window参数建议为奇数")

    df_copy = df.copy()

    # 计算过滤指标
    if filter_type == 'rsi':
        df_copy['filter_indicator'] = calculate_RSI(df, 14)
        if label_type == 'up':
            # 如果没有提供阈值，使用默认值
            if threshold is None:
                threshold = 30
            filter_condition = df_copy['filter_indicator'] < threshold
        else:  # down
            if threshold is None:
                threshold = 70
            filter_condition = df_copy['filter_indicator'] > threshold
    else:  # cti
        df_copy['filter_indicator'] = calculate_fast_cti(df)
        if label_type == 'up':
            if threshold is None:
                threshold = -0.5
            filter_condition = df_copy['filter_indicator'] < threshold
        else:  # down
            if threshold is None:
                threshold = 0.5
            filter_condition = df_copy['filter_indicator'] > threshold

    df_copy['Label'] = np.nan

    # 预先计算涨跌标签
    rise_tag = np.full(len(df_copy), np.nan)
    for i in range(len(df_copy)):
        if i + look_forward < len(df_copy):
            close_now = df_copy['close'].iloc[i]
            close_future = df_copy['close'].iloc[i + look_forward]
            if label_type == 'up':
                rise_tag[i] = 1 if (close_future - close_now) > 0 else 0
            else:  # down
                rise_tag[i] = 1 if (close_now - close_future) > 0 else 0
        else:
            rise_tag[i] = np.nan

    half_w = window // 2
    idx_list = df_copy.index.tolist()
    for i in range(len(df_copy)):
        index = idx_list[i]
        if filter_condition.iloc[i]:
            start = max(0, i - half_w)
            end = min(len(df_copy), i + half_w + 1)
            #start = max(0, i )
            #end = min(len(df_copy),i  + window ) 
            #start = max(0, i )
            #end = min(len(df_copy),i  + window ) 
            window_rise_tag = rise_tag[start:end]
            valid_count = np.sum(~np.isnan(window_rise_tag))
            if valid_count == 0:
                df_copy.loc[index, 'Label'] = np.nan
            else:
                ratio = np.nansum(window_rise_tag == 1) / valid_count
                df_copy.loc[index, 'Label'] = ratio
        else:
            df_copy.loc[index, 'Label'] = np.nan

    return df_copy['Label']

