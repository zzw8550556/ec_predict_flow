# EC Predict Flow

事件合约工作流Web应用

# 系统介绍

欢迎使用**事件合约预测工作流系统**！这是一个完整的量化交易工作流程平台，专注于事件合约的预测和回测。

## 系统概述

本系统提供了从数据获取到策略回测的完整工作流程，包括：

- 📊 **数据下载** - 从交易所下载K线数据并保存到本地
- 🔧 **特征计算** - 自带多种量价特征计算功能,计算特征因子并为涨跌事件设置真实值标签
- 🤖 **模型训练** - 使用LightGBM训练预测模型
- 📈 **模型解释** - 使用SHAP生成特征解释图
- 🔍 **模型分析** - 使用代理模型筛选关键特征,得到对应的判断阈值
- 📉 **构建回测** - 根据特征的判断阈值构建回测策略,并完成回测

## 工作流程

系统采用模块化设计，每个模块负责工作流程中的特定步骤。整个流程执行以下顺序：

1. 数据下载
2. 特征计算
3. 模型训练
4. 模型解释
5. 模型分析
6. 构建回测

## 技术栈

- **前端**: vite + Element Plus
- **后端**: Python + FastAPI
- **机器学习**: LightGBM + SHAP
- **数据处理**: Pandas + NumPy


## 安装和运行

### 后端设置

1. 安装依赖
```bash
cd backend
pip install -r requirements.txt
```

2. 配置环境变量
```bash
cp .env.example .env
# 编辑.env文件，填入必要的配置
```

3. 启动Redis
```bash
redis-server
```

4. 启动Celery Worker
```bash
cd backend
celery -A app.core.celery_app worker --loglevel=info --pool=solo
```

5. 启动FastAPI服务
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 前端设置

1. 安装依赖
```bash
cd frontend
npm install
```

2. 启动开发服务器
```bash
npm run dev
```

3. 访问应用
打开浏览器访问: http://localhost:5173

## API文档

启动后端服务后，访问以下地址查看API文档:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


