from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"


class DataDownloadRequest(BaseModel):
    symbol: str = "ETHUSDT"
    start_date: str
    end_date: str
    interval: str = "1m"
    proxy: Optional[str] = None


class FeatureCalculationRequest(BaseModel):
    data_file: str
    alpha_types: List[str] = ["alpha216"]  # 默认alpha216


class LabelCalculationRequest(BaseModel):
    data_file: str  # 原始数据文件名
    window: int = 29  # 标签窗口
    look_forward: int = 10  # 预测周期
    label_type: str = 'up'  # 标签类型：'up'上涨或'down'下跌
    filter_type: str = 'rsi'  # 过滤类型：'rsi'或'cti'
    threshold: Optional[float] = None  # 过滤阈值，如果为None则使用默认值


class LabelCalculationV2Request(BaseModel):
    data_file: str  # 原始数据文件名
    look_forward: int = 10  # 预测周期
    label_type: str = 'up'  # 标签类型：'up'上涨或'down'下跌
    filter_type: str = 'rsi'  # 过滤类型：'rsi'或'cti'
    threshold: Optional[float] = None  # 过滤阈值，如果为None则使用默认值
    methods: List[str] = ["safety_buffer"]  # 方法列表：可包含 'safety_buffer', 'average_price', 'multi_horizon'
    buffer_multiplier: float = 0.5  # 安全垫倍数（用于safety_buffer）
    avg_score_threshold: float = 0.0  # 平均分数阈值（用于average_price）


class ModelTrainingRequest(BaseModel):
    features_file: str  # 特征文件名
    labels_file: str  # 标签文件名
    num_boost_round: int = 500  # boosting迭代次数


class ModelInterpretationRequest(BaseModel):
    model_file: str  # 模型文件名


class ModelAnalysisRequest(BaseModel):
    model_file: str  # 模型文件名
    selected_features: List[str]  # 用户选择的关键特征（一般不多于8个）
    max_depth: int = 3  # 决策树最大深度
    min_samples_split: int = 100  # 节点分裂所需的最小样本数


class BacktestConstructionRequest(BaseModel):
    features_file: str  # 特征数据文件名（包含特征的已处理数据）
    decision_rules: List[Dict[str, Any]]  # 决策规则列表
    backtest_type: str = 'long'  # 回测类型：'long' 开多单回测，'short' 开空单回测
    filter_type: str = 'rsi'  # 过滤指标：'rsi' 或 'cti'
    look_forward_bars: int = 10  # 未来看多少根K线判断盈亏
    win_profit: float = 4.0  # 盈利金额
    loss_cost: float = 5.0  # 亏损金额
    initial_balance: float = 1000.0  # 初始余额


class BacktestExecutionRequest(BaseModel):
    strategy_config: Dict[str, Any]
    data_file: str


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class WorkflowStatus(BaseModel):
    workflow_id: str
    stages: Dict[str, TaskStatusResponse]
