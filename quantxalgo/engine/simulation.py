"""
Live Trading Simulation Engine
==============================

Simulates a live trading environment for the dashboard.
Maintains continuous state for:
- Portfolio NAV and holdings
- Active strategy performance
- Real-time market data
- Execution logs
"""

import asyncio
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List

from quantxalgo.portfolio.manager import PortfolioManager
from quantxalgo.portfolio.position import Position


class SimulationEngine:
    """Live trading simulation engine for dashboard connectivity."""
    
    def __init__(self, initial_capital: float = 10_000_000.0):
        self.initial_capital = initial_capital
        self.portfolio = PortfolioManager(initial_capital)
        
        # Active strategies with performance tracking
        self.strategies: List[Dict] = [
            {"name": "Alpha-Neutral-Eq", "type": "Market Neutral", "alloc": 0.25, "pnl": 0.0, "sharpe": 2.4, "dd": 0.0},
            {"name": "Global-Macro-FX", "type": "Global Macro", "alloc": 0.20, "pnl": 0.0, "sharpe": 1.8, "dd": 0.0},
            {"name": "Vol-Carry-SPX", "type": "Volatility", "alloc": 0.15, "pnl": 0.0, "sharpe": 3.1, "dd": 0.0},
            {"name": "Crypto-Mom-L1", "type": "Trend Following", "alloc": 0.15, "pnl": 0.0, "sharpe": 0.9, "dd": 0.0},
            {"name": "StatArb-Pairs-EU", "type": "Statistical Arb", "alloc": 0.25, "pnl": 0.0, "sharpe": 2.8, "dd": 0.0},
        ]
        
        # Market data feeds
        self.market_data: Dict[str, float] = {
            "SPY": 450.0, "QQQ": 380.0, "IWM": 200.0, 
            "AAPL": 180.0, "MSFT": 350.0, "BTC": 65000.0
        }
        
        # Historical NAV curve
        self.history: List[Dict] = []
        
        # Recent trades
        self.trades: List[Dict] = []
        
        # System logs
        self.logs: List[str] = []
        
        self.running = False
        self._last_tick = datetime.utcnow()
        
        # Initialize positions using Position objects
        self._init_positions()
        
        # Generate historical NAV data
        self._init_history()
    
    def _init_positions(self):
        """Initialize portfolio with starting positions."""
        for symbol, price in [("SPY", 450.0), ("QQQ", 380.0), ("AAPL", 180.0)]:
            pos = Position(symbol=symbol)
            qty = int(self.initial_capital * 0.1 / price)  # 10% allocation each
            pos.add(qty, price, datetime.utcnow())
            self.portfolio.positions[symbol] = pos
            self.portfolio.cash -= qty * price
    
    def _init_history(self):
        """Generate historical equity curve."""
        base_nav = self.portfolio.total_equity
        for i in range(90):
            dt = datetime.utcnow() - timedelta(days=90-i)
            val = base_nav * (1 + np.random.normal(0.0005, 0.01) * (90-i))
            self.history.append({"date": dt.isoformat(), "value": val})

    async def start(self):
        """Start the simulation loop."""
        self.running = True
        self._add_log("System", "INFO", "Trading Engine Started")
        asyncio.create_task(self._loop())

    async def stop(self):
        """Stop the simulation loop."""
        self.running = False
        self._add_log("System", "INFO", "Trading Engine Stopped")

    def _add_log(self, source: str, level: str, msg: str):
        """Add a log entry."""
        ts = datetime.utcnow().strftime("%H:%M:%S")
        self.logs.insert(0, f"[{ts}] [{source}] [{level}] {msg}")
        if len(self.logs) > 100:
            self.logs.pop()

    async def _loop(self):
        """Main simulation loop."""
        while self.running:
            await self._tick()
            await asyncio.sleep(1)

    async def _tick(self):
        """Process one simulation tick."""
        try:
            # 1. Update market prices (random walk)
            for symbol in self.market_data:
                change = np.random.normal(0, 0.0003)
                self.market_data[symbol] *= (1 + change)
            
            # 2. Update portfolio positions with new prices
            self.portfolio.update_prices(self.market_data)
            
            # 3. Record current NAV
            current_nav = self.portfolio.total_equity
            self._last_tick = datetime.utcnow()
            
            # Update history (every 5 minutes)
            if len(self.history) == 0 or (datetime.utcnow() - datetime.fromisoformat(self.history[-1]["date"])).seconds > 300:
                self.history.append({"date": datetime.utcnow().isoformat(), "value": current_nav})
                if len(self.history) > 500:
                    self.history.pop(0)

            # 4. Update strategy P&L
            for strat in self.strategies:
                pnl_change = np.random.normal(0, 50)
                strat["pnl"] += pnl_change
                
                # Track drawdown
                if strat["pnl"] < 0:
                    strat["dd"] = min(strat["dd"], strat["pnl"] / (self.initial_capital * strat["alloc"]))
            
            # 5. Random trade generation (5% chance per tick)
            if random.random() < 0.05:
                await self._execute_random_trade()

        except Exception as e:
            self._add_log("Engine", "ERROR", f"Tick error: {e}")

    async def _execute_random_trade(self):
        """Execute a random simulated trade."""
        symbol = random.choice(list(self.market_data.keys()))
        side = random.choice(["BUY", "SELL"])
        qty = random.randint(10, 100)
        price = self.market_data[symbol]
        
        # Record trade
        self.trades.insert(0, {
            "time": datetime.utcnow().strftime("%H:%M:%S.%f")[:-3],
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": round(price, 2),
            "value": round(qty * price, 2),
            "strategy": random.choice([s["name"] for s in self.strategies])
        })
        if len(self.trades) > 50:
            self.trades.pop()
        
        # Update position
        if symbol in self.portfolio.positions:
            pos = self.portfolio.positions[symbol]
            if side == "BUY":
                pos.add(qty, price, datetime.utcnow())
                self.portfolio.cash -= qty * price
            else:
                pos.add(-qty, price, datetime.utcnow())
                self.portfolio.cash += qty * price
        else:
            if side == "BUY":
                pos = Position(symbol=symbol)
                pos.add(qty, price, datetime.utcnow())
                self.portfolio.positions[symbol] = pos
                self.portfolio.cash -= qty * price
        
        self._add_log("Execution", "ORDER", f"FILLED: {side} {qty} {symbol} @ ${price:.2f}")

    def get_dashboard_state(self) -> Dict:
        """Get complete dashboard state for frontend."""
        current_nav = self.portfolio.total_equity
        prev_nav = self.history[-2]["value"] if len(self.history) > 1 else current_nav
        daily_pnl = current_nav - prev_nav
        
        # Calculate dynamic Sharpe
        returns = []
        for i in range(1, min(30, len(self.history))):
            ret = (self.history[-i]["value"] / self.history[-i-1]["value"]) - 1
            returns.append(ret)
        
        avg_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 0.01
        sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        
        return {
            "nav": current_nav,
            "prev_nav": prev_nav,
            "daily_pnl": daily_pnl,
            "sharpe": max(0.5, min(4.0, 2.45 + sharpe)),  # Bounded for display
            "var_95": current_nav * 0.015,
            "max_drawdown": self.portfolio.current_drawdown,
            "strategies": self.strategies,
            "equity_curve": self.history[-90:],  # Last 90 points
            "market_data": {k: round(v, 2) for k, v in self.market_data.items()},
            "trades": self.trades[:20],
            "logs": self.logs[:50],
            "positions": {s: {"qty": p.quantity, "value": p.market_value, "pnl": p.unrealized_pnl} 
                         for s, p in self.portfolio.positions.items()},
            "cash": self.portfolio.cash,
            "total_equity": current_nav,
            "leverage": self.portfolio.leverage,
        }
    
    def liquidate_all(self):
        """Emergency liquidate all positions."""
        for symbol, pos in list(self.portfolio.positions.items()):
            self.portfolio.cash += pos.market_value
            self._add_log("Risk", "CRITICAL", f"LIQUIDATED: {pos.quantity} {symbol}")
        self.portfolio.positions.clear()
        self._add_log("Risk", "CRITICAL", "All positions liquidated")
    
    def rebalance(self):
        """Trigger portfolio rebalance."""
        self._add_log("Portfolio", "INFO", "Rebalance triggered - reallocating capital")
        for s in self.strategies:
            s["alloc"] = 0.20  # Equal weight
        self._add_log("Portfolio", "INFO", "Rebalance complete - equal weight applied")
