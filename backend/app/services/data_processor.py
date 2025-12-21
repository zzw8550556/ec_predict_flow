# eth_data_processor.py
import pandas as pd
import numpy as np
# 引入您可能需要的技术指标库, 例如 talib
# import talib
import math
from tqdm import tqdm

class DataProcessor:
    def __init__(self, ohlcv_df, instrument_name='ETHUSDT'):
        """
        初始化数据处理器
        :param ohlcv_df: Pandas DataFrame, 包含ETH的OHLCV数据，
                         必须有 'datetime' (pd.Timestamp), 'open', 'high', 'low', 'close', 'volume' 列。
                         'datetime' 列应该是唯一的，并已排序。
        :param instrument_name: 字符串, 交易对的名称，如 'ETHUSDT'
        """
        self.df = ohlcv_df.copy()
        self.instrument_name = instrument_name
        self.feature_columns = []
        self.label_column_name = 'label_return' # LARA中的 self.out

        if not isinstance(self.df.index, pd.DatetimeIndex):
            if 'datetime' in self.df.columns:
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])
                self.df = self.df.set_index('datetime', drop=True)
            else:
                raise ValueError("DataFrame必须包含 'datetime' 列或拥有 DatetimeIndex.")
        self.df = self.df.sort_index()


    def generate_features_alpha158(self):
        """
        生成158个技术指标因子，计算方法基于Alpha158DL中的158因子公式。
        包含：
          1. kbar因子 (9个因子)
          2. Price因子（仅使用OPEN, HIGH, LOW, CLOSE; window=0，共4个因子）
          3. Rolling因子 (29组因子，每组对窗口[w]计算, w in [5,10,20,30,60]，共145个因子)
        """
        eps = 1e-12
        df = self.df.copy()

        # 1. KBar因子 (9个因子)
        df['feature_KMID']   = (df['close'] - df['open']) / df['open']
        df['feature_KLEN']   = (df['high'] - df['low']) / df['open']
        df['feature_KMID2']  = (df['close'] - df['open']) / (df['high'] - df['low'] + eps)
        df['feature_KUP']    = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['feature_KUP2']   = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + eps)
        df['feature_KLOW']   = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        df['feature_KLOW2']  = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + eps)
        df['feature_KSFT']   = (2 * df['close'] - df['high'] - df['low']) / df['open']
        df['feature_KSFT2']  = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + eps)

        # 2. Price因子 (仅使用OPEN, HIGH, LOW, CLOSE; window=0), 共4个因子
        for field in ['open', 'high', 'low', 'close']:
            col_name = f'feature_PRICE_{field.upper()}0'
            df[col_name] = df[field] / df['close']

        # 3. Rolling因子 (29组因子，每组在窗口 [5,10,20,30,60] 下计算，共145个因子)
        windows = [5, 10, 20, 30, 60]

        # 定义滚动线性回归函数，返回斜率、R平方和最后一个残差
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

        for w in windows:
            # ROC: 前w期的close与当前close比值
            df[f'feature_ROC{w}'] = df['close'].shift(w) / df['close']

            # MA: w期移动平均
            df[f'feature_MA{w}'] = df['close'].rolling(window=w).mean() / df['close']

            # STD: w期标准差
            df[f'feature_STD{w}'] = df['close'].rolling(window=w).std() / df['close']

            # BETA, RSQR, RESI：采用滚动线性回归计算
            df[f'feature_BETA{w}'] = df['close'].rolling(window=w).apply(lambda x: rolling_linreg(x)[0], raw=True) / df['close']
            df[f'feature_RSQR{w}'] = df['close'].rolling(window=w).apply(lambda x: rolling_linreg(x)[1], raw=True)
            df[f'feature_RESI{w}'] = df['close'].rolling(window=w).apply(lambda x: rolling_linreg(x)[2], raw=True) / df['close']

            # MAX: w期内最高价除以当前close
            df[f'feature_MAX{w}'] = df['high'].rolling(window=w).max() / df['close']

            # LOW: w期内最低价除以当前close
            df[f'feature_LOW{w}'] = df['low'].rolling(window=w).min() / df['close']

            # QTLU: w期内close的0.8分位数
            df[f'feature_QTLU{w}'] = df['close'].rolling(window=w).quantile(0.8) / df['close']

            # QTLD: w期内close的0.2分位数
            df[f'feature_QTLD{w}'] = df['close'].rolling(window=w).quantile(0.2) / df['close']

            # RANK: 当前close在w期内的百分位排名
            df[f'feature_RANK{w}'] = df['close'].rolling(window=w).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

            # RSV: (close - w期内最低低价) / (w期内最高价 - w期内最低低价)
            roll_low  = df['low'].rolling(window=w).min()
            roll_high = df['high'].rolling(window=w).max()
            df[f'feature_RSV{w}'] = (df['close'] - roll_low) / (roll_high - roll_low + eps)

            # IMAX: w期内最高价出现位置（归一化）
            df[f'feature_IMAX{w}'] = df['high'].rolling(window=w).apply(lambda x: np.argmax(x) / w, raw=True)

            # IMIN: w期内最低价出现位置（归一化）
            df[f'feature_IMIN{w}'] = df['low'].rolling(window=w).apply(lambda x: np.argmin(x) / w, raw=True)

            # IMXD: 最高价和最低价位置差（归一化）
            imax = df['high'].rolling(window=w).apply(lambda x: np.argmax(x), raw=True)
            imin = df['low'].rolling(window=w).apply(lambda x: np.argmin(x), raw=True)
            df[f'feature_IMXD{w}'] = (imax - imin) / w

            # CORR: w期内close与log(volume+1)的相关系数
            log_vol = np.log(df['volume'] + 1)
            df[f'feature_CORR{w}'] = df['close'].rolling(window=w).corr(log_vol)

            # CORD: w期内 (close/前一日close) 与 log( volume/前一日volume + 1) 的相关系数
            ret = df['close'] / df['close'].shift(1)
            vol_ret = np.log(df['volume'] / df['volume'].shift(1) + 1)
            df[f'feature_CORD{w}'] = ret.rolling(window=w).corr(vol_ret)

            # CNTP: w期内上涨比例
            df[f'feature_CNTP{w}'] = df['close'].diff().rolling(window=w).apply(lambda x: np.mean(x > 0))

            # CNTN: w期内下跌比例
            df[f'feature_CNTN{w}'] = df['close'].diff().rolling(window=w).apply(lambda x: np.mean(x < 0))

            # CNTD: 上涨比例与下跌比例之差
            df[f'feature_CNTD{w}'] = df[f'feature_CNTP{w}'] - df[f'feature_CNTN{w}']

            # SUMP: 上涨天数占比 (正差的天数/绝对变化之和)
            def sump(x):
                pos = np.sum(x > 0)
                denom = np.sum(np.abs(x))
                return pos / (denom + eps)
            df[f'feature_SUMP{w}'] = df['close'].diff().rolling(window=w).apply(sump, raw=True)

            # SUMN: 下跌天数占比 (负差的天数/绝对变化之和)
            def sumn(x):
                neg = np.sum(x < 0)
                denom = np.sum(np.abs(x))
                return neg / (denom + eps)
            df[f'feature_SUMN{w}'] = df['close'].diff().rolling(window=w).apply(sumn, raw=True)

            # SUMD: SUMP与SUMN的差值
            df[f'feature_SUMD{w}'] = df[f'feature_SUMP{w}'] - df[f'feature_SUMN{w}']

            # VMA: w期内成交量均值与当前成交量比值
            df[f'feature_VMA{w}'] = df['volume'].rolling(window=w).mean() / (df['volume'] + eps)

            # VSTD: w期内成交量标准差与当前成交量比值
            df[f'feature_VSTD{w}'] = df['volume'].rolling(window=w).std() / (df['volume'] + eps)

            # WVMA: 利用成交量加权的波动率指标
            vol_factor = abs(df['close'] / df['close'].shift(1) - 1) * df['volume']
            df[f'feature_WVMA{w}'] = vol_factor.rolling(window=w).std() / (vol_factor.rolling(window=w).mean() + eps)

            # VSUMP: w期内成交量上升日占比
            def vsump(x):
                pos = np.sum(x > 0)
                denom = np.sum(np.abs(x))
                return pos / (denom + eps)
            df[f'feature_VSUMP{w}'] = df['volume'].diff().rolling(window=w).apply(vsump, raw=True)

            # VSUMN: w期内成交量下降日占比
            def vsumn(x):
                neg = np.sum(x < 0)
                denom = np.sum(np.abs(x))
                return neg / (denom + eps)
            df[f'feature_VSUMN{w}'] = df['volume'].diff().rolling(window=w).apply(vsumn, raw=True)

            # VSUMD: VSUMP与VSUMN的差值
            df[f'feature_VSUMD{w}'] = df[f'feature_VSUMP{w}'] - df[f'feature_VSUMN{w}']

        # 记录所有生成的特征列名
        self.feature_columns = [col for col in df.columns if col.startswith('feature_')]
        print(f"Generated {len(self.feature_columns)} features.")
        self.df = df
        return self.df
    
    def generate_features_alpha216(self):
        """
        生成245个技术指标因子，计算方法基于Alpha245DL中的245因子公式。
        包含：
          1. kbar因子 (9个因子)
          2. Price因子（仅使用OPEN, HIGH, LOW, CLOSE; window=0，共4个因子）
          3. Rolling因子 (29组因子，每组对窗口[w]计算, w in [5,10,20,30,60]，共145个因子)
        """
        eps = 1e-12
        df = self.df.copy()

        # 1. KBar因子 (9个因子)
        df['feature_KMID']   = (df['close'] - df['open']) / df['open']
        df['feature_KLEN']   = (df['high'] - df['low']) / df['open']
        df['feature_KMID2']  = (df['close'] - df['open']) / (df['high'] - df['low'] + eps)
        df['feature_KUP']    = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['feature_KUP2']   = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + eps)
        df['feature_KLOW']   = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        df['feature_KLOW2']  = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + eps)
        df['feature_KSFT']   = (2 * df['close'] - df['high'] - df['low']) / df['open']
        df['feature_KSFT2']  = (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low'] + eps)

        # 2. Price因子 (仅使用OPEN, HIGH, LOW, CLOSE; window=0), 共4个因子
        for field in ['open', 'high', 'low', 'close']:
            col_name = f'feature_PRICE_{field.upper()}0'
            df[col_name] = df[field] / df['close']

        # 3. Rolling因子 (29组因子，每组在窗口 [5,10,20,30,60] 下计算，共145个因子)
        windows = [5, 10, 20, 30, 60 , 120, 240]

        # 定义滚动线性回归函数，返回斜率、R平方和最后一个残差
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

        for w in windows:
            # ROC: 前w期的close与当前close比值
            df[f'feature_ROC{w}'] = df['close'].shift(w) / df['close']

            # MA: w期移动平均
            df[f'feature_MA{w}'] = df['close'].rolling(window=w).mean() / df['close']

            # STD: w期标准差
            df[f'feature_STD{w}'] = df['close'].rolling(window=w).std() / df['close']

            # BETA, RSQR, RESI：采用滚动线性回归计算
            df[f'feature_BETA{w}'] = df['close'].rolling(window=w).apply(lambda x: rolling_linreg(x)[0], raw=True) / df['close']
            df[f'feature_RSQR{w}'] = df['close'].rolling(window=w).apply(lambda x: rolling_linreg(x)[1], raw=True)
            df[f'feature_RESI{w}'] = df['close'].rolling(window=w).apply(lambda x: rolling_linreg(x)[2], raw=True) / df['close']

            # MAX: w期内最高价除以当前close
            df[f'feature_MAX{w}'] = df['high'].rolling(window=w).max() / df['close']

            # LOW: w期内最低价除以当前close
            df[f'feature_LOW{w}'] = df['low'].rolling(window=w).min() / df['close']

            # QTLU: w期内close的0.8分位数
            df[f'feature_QTLU{w}'] = df['close'].rolling(window=w).quantile(0.8) / df['close']

            # QTLD: w期内close的0.2分位数
            df[f'feature_QTLD{w}'] = df['close'].rolling(window=w).quantile(0.2) / df['close']

            # RANK: 当前close在w期内的百分位排名
            df[f'feature_RANK{w}'] = df['close'].rolling(window=w).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)

            # RSV: (close - w期内最低低价) / (w期内最高价 - w期内最低低价)
            roll_low  = df['low'].rolling(window=w).min()
            roll_high = df['high'].rolling(window=w).max()
            df[f'feature_RSV{w}'] = (df['close'] - roll_low) / (roll_high - roll_low + eps)

            # IMAX: w期内最高价出现位置（归一化）
            df[f'feature_IMAX{w}'] = df['high'].rolling(window=w).apply(lambda x: np.argmax(x) / w, raw=True)

            # IMIN: w期内最低价出现位置（归一化）
            df[f'feature_IMIN{w}'] = df['low'].rolling(window=w).apply(lambda x: np.argmin(x) / w, raw=True)

            # IMXD: 最高价和最低价位置差（归一化）
            imax = df['high'].rolling(window=w).apply(lambda x: np.argmax(x), raw=True)
            imin = df['low'].rolling(window=w).apply(lambda x: np.argmin(x), raw=True)
            df[f'feature_IMXD{w}'] = (imax - imin) / w

            # CORR: w期内close与log(volume+1)的相关系数
            log_vol = np.log(df['volume'] + 1)
            df[f'feature_CORR{w}'] = df['close'].rolling(window=w).corr(log_vol)

            # CORD: w期内 (close/前一日close) 与 log( volume/前一日volume + 1) 的相关系数
            ret = df['close'] / df['close'].shift(1)
            vol_ret = np.log(df['volume'] / df['volume'].shift(1) + 1)
            df[f'feature_CORD{w}'] = ret.rolling(window=w).corr(vol_ret)

            # CNTP: w期内上涨比例
            df[f'feature_CNTP{w}'] = df['close'].diff().rolling(window=w).apply(lambda x: np.mean(x > 0))

            # CNTN: w期内下跌比例
            df[f'feature_CNTN{w}'] = df['close'].diff().rolling(window=w).apply(lambda x: np.mean(x < 0))

            # CNTD: 上涨比例与下跌比例之差
            df[f'feature_CNTD{w}'] = df[f'feature_CNTP{w}'] - df[f'feature_CNTN{w}']

            # SUMP: 上涨天数占比 (正差的天数/绝对变化之和)
            def sump(x):
                pos = np.sum(x > 0)
                denom = np.sum(np.abs(x))
                return pos / (denom + eps)
            df[f'feature_SUMP{w}'] = df['close'].diff().rolling(window=w).apply(sump, raw=True)

            # SUMN: 下跌天数占比 (负差的天数/绝对变化之和)
            def sumn(x):
                neg = np.sum(x < 0)
                denom = np.sum(np.abs(x))
                return neg / (denom + eps)
            df[f'feature_SUMN{w}'] = df['close'].diff().rolling(window=w).apply(sumn, raw=True)

            # SUMD: SUMP与SUMN的差值
            df[f'feature_SUMD{w}'] = df[f'feature_SUMP{w}'] - df[f'feature_SUMN{w}']

            # VMA: w期内成交量均值与当前成交量比值
            df[f'feature_VMA{w}'] = df['volume'].rolling(window=w).mean() / (df['volume'] + eps)

            # VSTD: w期内成交量标准差与当前成交量比值
            df[f'feature_VSTD{w}'] = df['volume'].rolling(window=w).std() / (df['volume'] + eps)

            # WVMA: 利用成交量加权的波动率指标
            vol_factor = abs(df['close'] / df['close'].shift(1) - 1) * df['volume']
            df[f'feature_WVMA{w}'] = vol_factor.rolling(window=w).std() / (vol_factor.rolling(window=w).mean() + eps)

            # VSUMP: w期内成交量上升日占比
            def vsump(x):
                pos = np.sum(x > 0)
                denom = np.sum(np.abs(x))
                return pos / (denom + eps)
            df[f'feature_VSUMP{w}'] = df['volume'].diff().rolling(window=w).apply(vsump, raw=True)

            # VSUMN: w期内成交量下降日占比
            def vsumn(x):
                neg = np.sum(x < 0)
                denom = np.sum(np.abs(x))
                return neg / (denom + eps)
            df[f'feature_VSUMN{w}'] = df['volume'].diff().rolling(window=w).apply(vsumn, raw=True)

            # VSUMD: VSUMP与VSUMN的差值
            df[f'feature_VSUMD{w}'] = df[f'feature_VSUMP{w}'] - df[f'feature_VSUMN{w}']

        # 记录所有生成的特征列名
        self.feature_columns = [col for col in df.columns if col.startswith('feature_')]
        print(f"Generated {len(self.feature_columns)} features.")
        self.df = df
        return self.df
    
    def generate_features_alpha360(self):
        """
        根据Alpha360DL里面360个因子的计算方法生成特征因子，共360个因子。
        每组因子的计算方法如下:
          CLOSE系列: (过去lag期的close)/(当前close)，lag取值59~1，lag=0时为 $close/$close
          OPEN系列: (过去lag期的open)/(当前close)
          HIGH系列: (过去lag期的high)/(当前close)
          LOW系列: (过去lag期的low)/(当前close)
          VWAP系列: (过去lag期的vwap)/(当前close)，如果vwap不存在，则用 (high+low+close)/3
          VOLUME系列: (过去lag期的volume)/(volume+1e-12)，lag=0时为 $volume/($volume+1e-12)
        """
        df = self.df.copy()
        # 如果vwap不存在，则计算vwap = (high + low + close) / 3
        if 'vwap' not in self.df.columns:
            # 使用滚动窗口方法计算vwap，算法参考 calculate_vwap
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            length = 20  # 默认窗口长度，你可以按需调整
            pv = typical_price * self.df['volume']
            rolling_pv = pv.rolling(window=length).sum()
            rolling_vol = self.df['volume'].rolling(window=length).sum()
            self.df['vwap'] = rolling_pv / rolling_vol
            print(f"Column 'vwap' not found. Calculated as rolling VWAP with window={length}.")

        # CLOSE系列
        for lag in range(59, 0, -1):
            df[f'feature_CLOSE{lag}'] = df['close'].shift(lag) / df['close']
        df['feature_CLOSE0'] = df['close'] / df['close']

        # OPEN系列
        for lag in range(59, 0, -1):
            df[f'feature_OPEN{lag}'] = df['open'].shift(lag) / df['close']
        df['feature_OPEN0'] = df['open'] / df['close']

        # HIGH系列
        for lag in range(59, 0, -1):
            df[f'feature_HIGH{lag}'] = df['high'].shift(lag) / df['close']
        df['feature_HIGH0'] = df['high'] / df['close']

        # LOW系列
        for lag in range(59, 0, -1):
            df[f'feature_LOW{lag}'] = df['low'].shift(lag) / df['close']
        df['feature_LOW0'] = df['low'] / df['close']

        # VWAP系列
        for lag in range(59, 0, -1):
            df[f'feature_VWAP{lag}'] = df['vwap'].shift(lag) / df['close']
        df['feature_VWAP0'] = df['vwap'] / df['close']

        # VOLUME系列
        for lag in range(59, 0, -1):
            df[f'feature_VOLUME{lag}'] = df['volume'].shift(lag) / (df['volume'] + 1e-12)
        df['feature_VOLUME0'] = df['volume'] / (df['volume'] + 1e-12)

        # 更新特征列并返回处理后的df
        self.feature_columns = [col for col in df.columns if col.startswith('feature_')]
        print(f"Generated {len(self.feature_columns)} features.")
        self.df = df
        return self.df
    
    def generate_features_alpha101(self):
        """
        生成101个Alpha因子特征，计算方法基于 alphas101.py 中定义的所有 alpha101 因子公式。
        注意：本函数假定输入数据（self.df）只有基本的 OHLCV 数据（open, high, low, close, volume）。
        若缺少因子计算所需的其他数据列，则进行如下适应性处理：
          - 若无 "vwap" 列，则以 (high + low + close) / 3 填充；
        构造完成后，利用 alphas101.Alphas101 类计算所有 101 个因子，
        并将因子结果以新列形式加入 self.df，列名格式为 "feature_alpha_XXX"，同时更新 self.feature_columns。
        """
        # 适应性补充缺失的列
        if 'vwap' not in self.df.columns:
            # 使用滚动窗口方法计算vwap，算法参考 calculate_vwap
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            length = 20  # 默认窗口长度，你可以按需调整
            pv = typical_price * self.df['volume']
            rolling_pv = pv.rolling(window=length).sum()
            rolling_vol = self.df['volume'].rolling(window=length).sum()
            self.df['vwap'] = rolling_pv / rolling_vol
            print(f"Column 'vwap' not found. Calculated as rolling VWAP with window={length}.")
        
        # 使用基本的 OHLCV 来计算alpha101，不涉及市值、市场基准等数据
        
        # 动态导入 alphas101 模块中的 Alphas101 类（确保 alphas101.py 在 PYTHONPATH 中）
        from alphas101 import Alphas101
 
        factor_names = []
        # 构造 Alphas101 对象，传入当前 DataFrame
        alphas = Alphas101(self.df)
        for i in tqdm(range(1, 102), desc="计算 alpha101 特征"):
            func_name = f'alpha{i:03d}'
            feature_name = f'feature_{func_name}'
            try:
                func = getattr(alphas, func_name)
                result = func()
                self.df[feature_name] = result
                factor_names.append(feature_name)
            except Exception as e:
                print(f"Could not compute {func_name} due to error: {e}")
        
        self.feature_columns.extend(factor_names)
        print(f"Generated {len(factor_names)} alpha101 features.")
        return self.df
        
    
    def generate_features_alpha191(self):
        """
        生成191个Alpha因子特征，计算方法基于alps191.py中定义的所有alpha因子公式。
        需要确保 self.df 包含必要的列：'open', 'high', 'low', 'close', 'volume', 
        'vwap', 'amount', 'benchmark_open', 'benchmark_close'（其中部分因子用到 benchmark 数据）。
        该函数依赖于 alphas191.py 中的 Alphas191 类。计算过程中若遇异常，则打印出错误信息并跳过该因子计算。
        最后将所有计算出的特征名添加到 self.feature_columns 中，并返回处理后的 DataFrame。
        """
        # 若缺少必要的列，则进行数据补充
        # 适应性补充缺失的列
        if 'vwap' not in self.df.columns:
            # 使用滚动窗口方法计算vwap，算法参考 calculate_vwap
            typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
            length = 20  # 默认窗口长度，你可以按需调整
            pv = typical_price * self.df['volume']
            rolling_pv = pv.rolling(window=length).sum()
            rolling_vol = self.df['volume'].rolling(window=length).sum()
            self.df['vwap'] = rolling_pv / rolling_vol
            print(f"Column 'vwap' not found. Calculated as rolling VWAP with window={length}.")
        if 'amount' not in self.df.columns:
            self.df['amount'] = self.df['volume'] * self.df['close']
            print("Column 'amount' not found. Created as volume * close.")
        if 'benchmark_open' not in self.df.columns:
            self.df['benchmark_open'] = self.df['open']
            print("Column 'benchmark_open' not found. Substitute with 'open'.")
        if 'benchmark_close' not in self.df.columns:
            self.df['benchmark_close'] = self.df['close']
            print("Column 'benchmark_close' not found. Substitute with 'close'.")

        from alphas191 import Alphas191  # 确保 alphas191.py 在 Python 模块搜索路径中
 
        factor_names = []
        # 根据 self.df 构造 Alphas191 对象，algs 里的方法会直接基于 DataFrame 计算
        alphas = Alphas191(self.df)
        for i in tqdm(range(1, 192), desc="计算 alpha191 特征"):
            func_name = f'alpha{i:03d}'
            feature_name = f'feature_{func_name}'
            try:
                # 动态调用 Alphas191 内的方法计算因子
                func = getattr(alphas, func_name)
                result = func()
                # 将计算结果赋给 self.df 新列
                self.df[feature_name] = result
                factor_names.append(feature_name)
            except Exception as e:
                print(f"Could not compute {func_name} due to error: {e}")
        # 记录本次生成的所有 alpha191 特征列
        self.feature_columns.extend(factor_names)
        print(f"Generated {len(factor_names)} alpha191 features.")
        return self.df
    
    def generate_features_alpha_ch(self):
        """
        根据alpha_ch.py的featrues池和对应的计算函数，批量生成中文特征。
        """
        import alpha_ch
        import re

        df = self.df.copy()
        factor_names = []
        raw_fields = ['open', 'high', 'low', 'close', 'volume']  # 直接从原df复制
        fea_list = getattr(alpha_ch, 'featrues', [])
        for feat in fea_list:
            feat_stripped = feat.replace(' ', '').replace('-', '_').replace('.', '_').replace('=', '').replace('#', '').strip('\'"')
            try:
                # 1. 直接原始字段
                if feat in raw_fields:
                    df[f'feature_{feat}'] = df[feat]
                    factor_names.append(f'feature_{feat}')
                    continue

                # 2. 明确函数名
                func = None
                # 判断_参数 结尾，如 xxx_20
                match = re.match(r"([a-zA-Z0-9]+)_([0-9]+)$", feat_stripped)
                if match:
                    base, param = match.group(1), int(match.group(2))
                    func_name = f'calculate_{base}_{param}'
                    if hasattr(alpha_ch, func_name):
                        func = getattr(alpha_ch, func_name)
                        result = func(df, timeperiod=param)
                        df[f'feature_{feat}'] = result
                        factor_names.append(f'feature_{feat}')
                        continue
                # 含diff、slope等复杂后缀
                func_name = f'calculate_{feat_stripped}'
                if hasattr(alpha_ch, func_name):
                    func = getattr(alpha_ch, func_name)
                    result = func(df)
                    df[f'feature_{feat}'] = result
                    factor_names.append(f'feature_{feat}')
                    continue

                # fallback 尝试不带参数
                base_func_name = f'calculate_{feat_stripped.split("_")[0]}'
                if hasattr(alpha_ch, base_func_name):
                    func = getattr(alpha_ch, base_func_name)
                    result = func(df)
                    df[f'feature_{feat}'] = result
                    factor_names.append(f'feature_{feat}')
                    continue

                print(f"[Warning] 未找到对应的函数实现或逻辑: {feat}")
            except Exception as e:
                print(f"[Error] {feat}: {e}")
                continue

        self.feature_columns.extend(factor_names)
        print(f"Generated {len(factor_names)} alpha_ch特征.")
        self.df = df
        return self.df
    
    def generate_features_potato(self):
        """
        根据特征工程方法生成特征因子。
        包含价格特征、成交量特征、技术指标、统计特征等多个维度的特征。
        """

        df = self.df.copy()

        # 配置参数
        very_short_periods = [1, 2, 3, 5]
        short_term_periods = [5, 8, 13, 21]
        medium_term_periods = [21, 34, 55, 89]
        ultra_short_periods = [1, 2, 3]
        long_term_periods = [55, 89, 144, 233]

        # ============ 价格特征 ============
        # 1. 短期动量特征
        for period in very_short_periods + short_term_periods:
            if period <= len(df):
                df[f'feature_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # 2. 中期趋势特征
        for period in medium_term_periods:
            if period <= len(df):
                df[f'feature_trend_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)

        # 3. 价格加速度
        if 'feature_momentum_13' in df.columns and 'feature_momentum_5' in df.columns:
            df['feature_acceleration_5_13'] = df['feature_momentum_13'] - df['feature_momentum_5']
        if 'feature_momentum_34' in df.columns and 'feature_momentum_13' in df.columns:
            df['feature_acceleration_13_34'] = df['feature_momentum_34'] - df['feature_momentum_13']

        # 4. 价格效率
        if 'feature_momentum_21' in df.columns:
            df['feature_price_efficiency_21'] = abs(df['feature_momentum_21']) / (df['close'].rolling(21).std() + 1e-9)

        # ============ 成交量特征 ============
        # 5. 成交量Z-score
        df['feature_volume_zscore_30'] = (df['volume'] - df['volume'].rolling(30).mean()) / (df['volume'].rolling(30).std() + 1e-9)
        df['feature_volume_zscore_89'] = (df['volume'] - df['volume'].rolling(89).mean()) / (df['volume'].rolling(89).std() + 1e-9)

        # 6. 成交量变化率
        df['feature_volume_change_5'] = df['volume'].pct_change(5)
        df['feature_volume_change_21'] = df['volume'].pct_change(21)

        # 7. 量价关系
        df['feature_price_volume_corr_21'] = df['close'].rolling(21).corr(df['volume'])

        # ============ 技术指标 ============
        # 8. 移动平均
        for period in [8, 21, 55, 144]:
            if period <= len(df):
                df[f'feature_sma_{period}'] = df['close'].rolling(period).mean()
                df[f'feature_ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                df[f'feature_price_vs_sma_{period}'] = (df['close'] - df[f'feature_sma_{period}']) / df[f'feature_sma_{period}']

        # 9. RSI（多周期）
        for period in [6, 14, 28]:
            df[f'feature_rsi_{period}'] = self._calculate_rsi(df['close'], period)

        # 10. MACD
        df['feature_ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['feature_ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['feature_macd'] = df['feature_ema_12'] - df['feature_ema_26']
        df['feature_macd_signal'] = df['feature_macd'].ewm(span=9, adjust=False).mean()
        df['feature_macd_hist'] = df['feature_macd'] - df['feature_macd_signal']

        # 11. 布林带（多周期）
        for period in [20, 55, 89]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'feature_bb_upper_{period}'] = sma + 2 * std
            df[f'feature_bb_lower_{period}'] = sma - 2 * std
            df[f'feature_bb_width_{period}'] = (df[f'feature_bb_upper_{period}'] - df[f'feature_bb_lower_{period}']) / sma
            df[f'feature_bb_position_{period}'] = (df['close'] - df[f'feature_bb_lower_{period}']) / (4 * std + 1e-9)

        # 12. ATR（平均真实波幅）
        for period in [14, 28]:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'feature_atr_{period}'] = tr.rolling(period).mean()
            df[f'feature_atr_ratio_{period}'] = df[f'feature_atr_{period}'] / df['close']

        # 13. 波动率特征
        for window in [21, 55, 144]:
            df[f'feature_volatility_{window}'] = df['close'].pct_change().rolling(window).std() * math.sqrt(365*24*60)

        # 14. 统计特征
        for window in [21, 55, 144]:
            df[f'feature_zscore_{window}'] = (df['close'] - df['close'].rolling(window).mean()) / (df['close'].rolling(window).std() + 1e-9)
            df[f'feature_skew_{window}'] = df['close'].rolling(window).skew()
            df[f'feature_kurt_{window}'] = df['close'].rolling(window).kurt()

        # 15. 支撑阻力特征
        for window in long_term_periods:
            df[f'feature_high_{window}'] = df['high'].rolling(window).max()
            df[f'feature_low_{window}'] = df['low'].rolling(window).min()
            df[f'feature_high_distance_{window}'] = (df[f'feature_high_{window}'] - df['close']) / df['close']
            df[f'feature_low_distance_{window}'] = (df['close'] - df[f'feature_low_{window}']) / df['close']

        # 16. K线形态特征
        df['feature_body_size'] = abs(df['close'] - df['open'])
        df['feature_upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['feature_lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['feature_body_ratio'] = df['feature_body_size'] / (df['high'] - df['low'] + 1e-9)
        df['feature_upper_shadow_ratio'] = df['feature_upper_shadow'] / (df['high'] - df['low'] + 1e-9)
        df['feature_lower_shadow_ratio'] = df['feature_lower_shadow'] / (df['high'] - df['low'] + 1e-9)

        # 17. 时间特征
        if isinstance(df.index, pd.DatetimeIndex):
            df['feature_hour'] = df.index.hour
            df['feature_minute'] = df.index.minute
            df['feature_day_of_week'] = df.index.dayofweek
            df['feature_hour_sin'] = np.sin(2 * np.pi * df['feature_hour'] / 24)
            df['feature_hour_cos'] = np.cos(2 * np.pi * df['feature_hour'] / 24)
            df['feature_day_sin'] = np.sin(2 * np.pi * df['feature_day_of_week'] / 7)
            df['feature_day_cos'] = np.cos(2 * np.pi * df['feature_day_of_week'] / 7)

        # 18. 组合特征
        # 趋势+成交量确认
        if 'feature_momentum_21' in df.columns and 'feature_volume_zscore_30' in df.columns:
            df['feature_trend_volume_21_30'] = ((df['feature_momentum_21'] > 0) & (df['feature_volume_zscore_30'] > 1)).astype(int)

        # 突破特征
        if 'feature_high_55' in df.columns and 'feature_low_55' in df.columns:
            df['feature_near_high_55'] = ((df['feature_high_55'] - df['close']) / df['close'] < 0.005).astype(int)
            df['feature_near_low_55'] = ((df['close'] - df['feature_low_55']) / df['close'] < 0.005).astype(int)

        # 多时间框架一致性
        for p1, p2 in [(8, 21), (21, 55), (55, 144)]:
            if f'feature_sma_{p1}' in df.columns and f'feature_sma_{p2}' in df.columns:
                df[f'feature_trend_alignment_{p1}_{p2}'] = ((df[f'feature_sma_{p1}'] > df[f'feature_sma_{p2}']).astype(int) * 2 - 1)

        # ============ 高频特征（10分钟专用） ============
        # 19. 极短期动量特征
        for period in ultra_short_periods:
            if period <= len(df):
                df[f'feature_ultra_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # 20. 价格冲击
        df['feature_price_impact'] = (df['high'] - df['low']) / df['close'] * 100

        # 21. 高频成交量特征
        df['feature_volume_zscore_10'] = (df['volume'] - df['volume'].rolling(10).mean()) / (df['volume'].rolling(10).std() + 1e-9)
        df['feature_volume_zscore_60'] = (df['volume'] - df['volume'].rolling(60).mean()) / (df['volume'].rolling(60).std() + 1e-9)
        df['feature_volume_change_3'] = df['volume'].pct_change(3)
        df['feature_volume_change_10'] = df['volume'].pct_change(10)

        # 22. 高频量价关系
        df['feature_price_volume_corr_10'] = df['close'].rolling(10).corr(df['volume'])
        df['feature_price_volume_corr_30'] = df['close'].rolling(30).corr(df['volume'])

        # 23. 高频移动平均
        for period in [3, 5, 10, 20, 30]:
            if period <= len(df):
                if f'feature_sma_{period}' not in df.columns:
                    df[f'feature_sma_{period}'] = df['close'].rolling(period).mean()
                if f'feature_ema_{period}' not in df.columns:
                    df[f'feature_ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                if f'feature_price_vs_sma_{period}' not in df.columns:
                    df[f'feature_price_vs_sma_{period}'] = (df['close'] - df[f'feature_sma_{period}']) / df[f'feature_sma_{period}']

        # 24. 高频RSI
        for period in [3, 6]:
            if f'feature_rsi_{period}' not in df.columns:
                df[f'feature_rsi_{period}'] = self._calculate_rsi(df['close'], period)

        # 25. 快速MACD
        df['feature_ema_6'] = df['close'].ewm(span=6, adjust=False).mean()
        df['feature_ema_13'] = df['close'].ewm(span=13, adjust=False).mean()
        df['feature_fast_macd'] = df['feature_ema_6'] - df['feature_ema_13']
        df['feature_fast_macd_signal'] = df['feature_fast_macd'].ewm(span=5, adjust=False).mean()
        df['feature_fast_macd_hist'] = df['feature_fast_macd'] - df['feature_fast_macd_signal']

        # 26. 高频布林带
        for period in [10, 20, 30]:
            if f'feature_bb_upper_{period}' not in df.columns:
                sma = df['close'].rolling(period).mean()
                std = df['close'].rolling(period).std()
                df[f'feature_bb_upper_{period}'] = sma + 2 * std
                df[f'feature_bb_lower_{period}'] = sma - 2 * std
                df[f'feature_bb_width_{period}'] = (df[f'feature_bb_upper_{period}'] - df[f'feature_bb_lower_{period}']) / sma
                df[f'feature_bb_position_{period}'] = (df['close'] - df[f'feature_bb_lower_{period}']) / (4 * std + 1e-9)

        # 27. 高频ATR
        for period in [5, 14]:
            if f'feature_atr_{period}' not in df.columns:
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift())
                low_close = abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df[f'feature_atr_{period}'] = tr.rolling(period).mean()
                df[f'feature_atr_ratio_{period}'] = df[f'feature_atr_{period}'] / df['close']

        # 28. 高频波动率
        for window in [5, 10, 20]:
            if f'feature_volatility_{window}' not in df.columns:
                df[f'feature_volatility_{window}'] = df['close'].pct_change().rolling(window).std() * math.sqrt(365*24*60)

        # 29. 高频统计特征
        for window in [10, 20, 30]:
            if f'feature_zscore_{window}' not in df.columns:
                df[f'feature_zscore_{window}'] = (df['close'] - df['close'].rolling(window).mean()) / (df['close'].rolling(window).std() + 1e-9)
            if f'feature_skew_{window}' not in df.columns:
                df[f'feature_skew_{window}'] = df['close'].rolling(window).skew()

        # 30. 高频支撑阻力
        for window in [30, 60]:
            df[f'feature_high_{window}'] = df['high'].rolling(window).max()
            df[f'feature_low_{window}'] = df['low'].rolling(window).min()
            df[f'feature_high_distance_{window}'] = (df[f'feature_high_{window}'] - df['close']) / df['close']
            df[f'feature_low_distance_{window}'] = (df['close'] - df[f'feature_low_{window}']) / df['close']

        # 31. 反转模式检测
        df['feature_hammer_pattern'] = ((df['feature_lower_shadow_ratio'] > 0.6) & (df['feature_body_ratio'] < 0.3)).astype(int)
        df['feature_shooting_star'] = ((df['feature_upper_shadow_ratio'] > 0.6) & (df['feature_body_ratio'] < 0.3)).astype(int)

        # 32. 高频组合特征
        if 'feature_momentum_5' in df.columns and 'feature_volume_zscore_10' in df.columns:
            df['feature_trend_volume_5_10'] = ((df['feature_momentum_5'] > 0) & (df['feature_volume_zscore_10'] > 1)).astype(int)

        if 'feature_high_30' in df.columns and 'feature_low_30' in df.columns:
            df['feature_near_high_30'] = ((df['feature_high_30'] - df['close']) / df['close'] < 0.003).astype(int)
            df['feature_near_low_30'] = ((df['close'] - df['feature_low_30']) / df['close'] < 0.003).astype(int)

        # 多时间框架一致性（高频）
        for p1, p2 in [(3, 10), (5, 20), (10, 30)]:
            if f'feature_sma_{p1}' in df.columns and f'feature_sma_{p2}' in df.columns:
                df[f'feature_trend_alignment_{p1}_{p2}'] = ((df[f'feature_sma_{p1}'] > df[f'feature_sma_{p2}']).astype(int) * 2 - 1)

        # 33. 价格变化趋势
        df['feature_price_trend_3'] = df['close'].diff(3) / df['close'].shift(3)
        df['feature_price_trend_5'] = df['close'].diff(5) / df['close'].shift(5)

        # 34. 成交量趋势
        df['feature_volume_trend_3'] = df['volume'].diff(3) / df['volume'].shift(3)
        df['feature_volume_trend_5'] = df['volume'].diff(5) / df['volume'].shift(5)

        # 35. 相对强弱
        if 'feature_momentum_5' in df.columns and 'feature_momentum_10' in df.columns:
            df['feature_relative_strength_5_10'] = df['feature_momentum_5'] - df['feature_momentum_10']

        # ============ 清理数据 ============
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        # 更新特征列并返回处理后的df
        self.feature_columns = [col for col in df.columns if col.startswith('feature_')]
        print(f"Generated {len(self.feature_columns)} potato features.")
        self.df = df
        return self.df
    
    def _calculate_rsi(self, series, period=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_labels(self, future_periods=20):
        """
        生成标签，即未来N期的收益率。
        :param future_periods: 预测未来多少个周期的收益率。
                               根据您的K线数据频率（如1小时、4小时、1天）来确定这个值。
        """
        self.label_column_name = f'label_fwd_return_{future_periods}'
        self.df[self.label_column_name] = (self.df['close'].shift(-future_periods) / self.df['close']) - 1
        print(f"Generated label: {self.label_column_name}")
        return self.df

    def get_processed_data(self):
        """
        清理数据，去除因滚动窗口或shift产生的NaN值。
        """
        self.df = self.df.dropna()
        # 确保所有特征列都是数值类型
        for col in self.feature_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna() # 再次dropna以防有转换错误
        return self.df
