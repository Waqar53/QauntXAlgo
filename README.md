# QuantXalgo

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

### 🏦 Institutional-Grade Algorithmic Trading Platform

**A full-stack hedge fund operating system with real Python code execution, live backtesting, and comprehensive risk management.**

[Features](#-features) • [Quick Start](#-quick-start) • [Quant Lab](#-quant-lab-python-ide) • [Architecture](#-architecture) • [Results](#-backtest-results)

</div>

---

## 🎯 What is QuantXalgo?

QuantXalgo is a **production-ready algorithmic trading platform** that simulates a real hedge fund's technology stack. Built for quants, traders, and fintech engineers who want to:

- ✅ **Write and test any Python trading algorithm** in a sandboxed execution environment
- ✅ **Backtest strategies** against historical data with realistic execution
- ✅ **Monitor real-time portfolio performance** with institutional-grade dashboards
- ✅ **Manage risk** with VaR, stress testing, and kill switch functionality

This is **not a toy project** — it's a complete hedge fund operating system.

---

## 🚀 Features

### 🧪 Quant Lab - Full Python IDE

Write **any trading algorithm** and execute it against market data:

```python
class TrendFollowingMomentum(QuantLabStrategy):
    def initialize(self):
        self.fast_ma = 10
        self.slow_ma = 30
        self.stop_loss = 0.05
        self.log("Strategy initialized")
        
    def on_bar(self, data, context):
        for symbol in self.symbols:
            df = data[symbol]
            fast = df["close"].rolling(self.fast_ma).mean()
            slow = df["close"].rolling(self.slow_ma).mean()
            
            if fast.iloc[-1] > slow.iloc[-1]:
                return [self.market_order(symbol, 100)]
        return []
```

**The code executes in real-time** with:
- 📊 Live equity curve visualization
- 📋 Complete trade log with P&L
- 📈 Performance metrics (Sharpe, Sortino, Max DD)
- 🔄 Position tracking and order management

### 📊 Live Dashboard

Real-time monitoring of:
- **NAV (Net Asset Value)** with historical equity curve
- **Daily P&L** across all strategies
- **Active positions** with unrealized gains
- **Risk metrics** (VaR, leverage, drawdown)

### 🛡️ Risk Management

- **Value at Risk (VaR)** - 95% confidence interval
- **Stress Testing** - 2008 Crisis, COVID-19, Flash Crash scenarios
- **Kill Switch** - Emergency position liquidation
- **Position Limits** - Concentration and leverage controls

### 📈 Reports Section

- **Performance Analytics** - YTD, MTD, inception returns
- **Trade History** - Full audit trail with filters
- **Strategy Attribution** - P&L breakdown by strategy
- **Risk Analytics** - Detailed risk factor analysis

---

## 💻 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/QuantXalgo.git
cd QuantXalgo

# Install dependencies
pip install -e .

# Start the API server
python -m uvicorn quantxalgo.api.main:app --host 0.0.0.0 --port 8000
```

### Launch the Dashboard

Open `frontend/index.html` in your browser and you'll see:

1. **Dashboard** - Real-time NAV and performance
2. **Quant Lab** - Write and test algorithms
3. **Portfolio** - Position management
4. **Reports** - Analytics and trade history
5. **Risk Mgmt** - VaR and stress testing

---

## 🧪 Quant Lab - Python IDE

### How It Works

1. **Write your strategy** extending `QuantLabStrategy`
2. **Configure parameters** (dates, capital, symbols)
3. **Click RUN BACKTEST**
4. **View results** - metrics, trades, equity curve

### Strategy Base Class

```python
class QuantLabStrategy(ABC):
    """Base class for all Quant Lab strategies."""
    
    def initialize(self):
        """Set up your parameters here."""
        pass
    
    def on_bar(self, data, context):
        """Called on each bar. Return list of Orders."""
        pass
    
    # Order helpers
    def market_order(symbol, quantity)
    def limit_order(symbol, quantity, price)
    def close_position(symbol)
    def order_target_percent(symbol, pct)
    
    # Utilities
    def log(message)  # Log to console
```

### Example: Bollinger Band Mean Reversion

```python
class BollingerMeanReversion(QuantLabStrategy):
    def initialize(self):
        self.period = 20
        self.std_dev = 2.0
        
    def on_bar(self, data, context):
        orders = []
        for symbol in self.symbols:
            df = data[symbol]
            if len(df) < self.period:
                continue
                
            sma = df["close"].rolling(self.period).mean()
            std = df["close"].rolling(self.period).std()
            upper = sma + (self.std_dev * std)
            lower = sma - (self.std_dev * std)
            price = df["close"].iloc[-1]
            
            if price < lower.iloc[-1]:
                orders.append(self.market_order(symbol, 100))
            elif price > upper.iloc[-1] and context.has_position(symbol):
                orders.append(self.close_position(symbol))
                
        return orders
```

---

## 📊 Backtest Results

### Trend Following Strategy

Tested on $25M capital over 3 years:

| Metric | Value |
|--------|-------|
| **Total Return** | +4.09% |
| **Absolute P&L** | +$1,023,537 |
| **Sharpe Ratio** | 0.81 |
| **Sortino Ratio** | 1.17 |
| **Max Drawdown** | -1.63% |
| **Total Trades** | 692 |
| **Execution Time** | 2.97s |

### Multi-Asset Allocation

```
📊 SYMBOLS TRADED:
   • NVDA: 16 trades, $1,578,181 volume
   • AAPL: 12 trades, $1,164,834 volume
   • MSFT: 20 trades, $1,048,603 volume  
   • GOOGL: 18 trades, $1,045,781 volume
   • AMZN: 10 trades, $1,002,463 volume
```

### Trade Execution Logs

```
📈 TREND ENTRY: GOOGL
   Price: $256.80
   MA Alignment: Fast > Slow > Trend ✓
   5d Momentum: +0.7%
   20d Momentum: +3.9%

⬆️ SCALING UP: AMZN
   Current P&L: +16.7%
   Adding 311 shares

💰 TAKING PROFIT: NVDA
   Gain: +20.5%
```

---

## 🏗️ Architecture

```
quantxalgo/
├── api/
│   └── main.py              # FastAPI server (50+ endpoints)
├── engine/
│   ├── backtest.py          # Event-driven backtester
│   ├── quant_lab.py         # Python code execution engine
│   └── simulation.py        # Live trading simulation
├── portfolio/
│   ├── manager.py           # Portfolio management
│   └── position.py          # Position tracking
├── strategies/
│   ├── base.py              # Strategy base class
│   ├── momentum/            # Trend following strategies
│   ├── mean_reversion/      # Mean reversion strategies
│   └── volatility/          # Volatility strategies
├── risk/
│   └── manager.py           # Risk management & VaR
└── core/
    ├── events.py            # Event system
    └── enums.py             # Order types, sides, etc.
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/live/dashboard` | GET | Real-time portfolio state |
| `/api/v1/quant-lab/execute` | POST | **Execute Python code** |
| `/api/v1/quant-lab/templates` | GET | Strategy templates |
| `/api/v1/backtest` | POST | Run backtest |
| `/api/v1/reports/performance` | GET | Performance analytics |
| `/api/v1/reports/trades` | GET | Trade history |
| `/api/v1/reports/risk` | GET | Risk metrics |
| `/api/v1/risk/scenarios` | GET | Stress test scenarios |

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, FastAPI, Pandas, NumPy |
| **API** | RESTful + WebSocket |
| **Frontend** | HTML5, JavaScript, Chart.js, CodeMirror |
| **Code Execution** | Sandboxed Python with AST validation |

---

## 🎓 Why This Project?

This project demonstrates:

1. **Full-Stack Development** - Backend API + Frontend dashboard
2. **Financial Engineering** - Portfolio theory, risk metrics, backtesting
3. **Systems Design** - Event-driven architecture, code sandboxing
4. **Production Thinking** - Error handling, logging, security

Built as a demonstration of enterprise-grade software engineering in quantitative finance.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built by Waqar Azim**

⭐ Star this repo if you found it impressive!

</div>
