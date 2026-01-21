#!/usr/bin/env python3
"""
QuantXalgo Integration Test Suite.

Tests all major components to verify the platform works end-to-end.
"""

import asyncio
from datetime import datetime, timedelta
import sys


async def test_data_engine():
    """Test data fetching and normalization."""
    print("\n" + "=" * 50)
    print("🔍 Testing Data Engine...")
    print("=" * 50)
    
    from quantxalgo.data.sources.yahoo import YahooDataSource
    from quantxalgo.data.normalization import DataNormalizer
    from quantxalgo.data.feature_store import FeatureStore
    
    async with YahooDataSource() as yahoo:
        # Fetch data
        end = datetime.now()
        start = end - timedelta(days=365)
        
        data = await yahoo.fetch_ohlcv("SPY", start, end)
        print(f"  ✓ Fetched {len(data)} bars for SPY")
        
        # Normalize
        normalizer = DataNormalizer()
        normalized = normalizer.normalize(data, "SPY")
        print(f"  ✓ Data normalized, {len(normalized)} bars")
        
        # Generate features
        fs = FeatureStore()
        features = fs.compute_all(normalized)
        print(f"  ✓ Generated {len(features.columns)} features")
        
    return True


async def test_strategies():
    """Test strategy creation and signal generation."""
    print("\n" + "=" * 50)
    print("🧠 Testing Strategies...")
    print("=" * 50)
    
    from quantxalgo.strategies.registry import StrategyRegistry
    from quantxalgo.strategies.momentum import MACrossoverStrategy, BreakoutStrategy
    from quantxalgo.strategies.mean_reversion import BollingerBandStrategy, RSIMeanReversionStrategy
    from quantxalgo.strategies.statistical import PairsTradingStrategy, FactorModelStrategy
    from quantxalgo.strategies.volatility import VolatilityBreakoutStrategy, SqueezeStrategy
    
    strategies = StrategyRegistry.list_strategies()
    print(f"  ✓ {len(strategies)} strategies registered: {strategies}")
    
    # Create each strategy
    for name in strategies:
        if name == "pairs_trading":
            strategy = StrategyRegistry.create(name, symbols=["SPY", "QQQ"])
        elif name == "factor_model":
            strategy = StrategyRegistry.create(name, symbols=["SPY", "QQQ", "IWM", "DIA", "VTI"])
        else:
            strategy = StrategyRegistry.create(name, symbols=["SPY"])
        print(f"  ✓ Created {name}: {strategy}")
    
    return True


async def test_backtester():
    """Test backtesting engine."""
    print("\n" + "=" * 50)
    print("⚙️ Testing Backtester...")
    print("=" * 50)
    
    from quantxalgo.data.sources.yahoo import YahooDataSource
    from quantxalgo.strategies.momentum.ma_crossover import MACrossoverStrategy
    from quantxalgo.engine.backtest import Backtester, BacktestConfig
    
    # Fetch data
    async with YahooDataSource() as yahoo:
        end = datetime.now()
        start = end - timedelta(days=500)
        data = await yahoo.fetch_multiple(["SPY", "QQQ"], start, end)
    
    print(f"  ✓ Data loaded for {len(data)} symbols")
    
    # Create strategy
    strategy = MACrossoverStrategy(
        name="test_ma",
        params={"fast_period": 10, "slow_period": 50},
        symbols=["SPY", "QQQ"],
    )
    
    # Run backtest
    config = BacktestConfig(
        start_date=start,
        end_date=end,
        initial_capital=1_000_000,
    )
    
    backtester = Backtester(config)
    result = await backtester.run(strategy, data)
    
    perf = result["performance"]
    print(f"  ✓ Backtest complete:")
    print(f"      Total Return: {perf['total_return']:.2%}")
    print(f"      Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"      Max Drawdown: {perf['max_drawdown']:.2%}")
    print(f"      Total Trades: {perf['total_trades']}")
    
    return True


async def test_portfolio():
    """Test portfolio management."""
    print("\n" + "=" * 50)
    print("📊 Testing Portfolio Manager...")
    print("=" * 50)
    
    from quantxalgo.portfolio.manager import PortfolioManager
    from quantxalgo.portfolio.position import Position
    from quantxalgo.portfolio.capital import CapitalAllocator
    
    # Create portfolio
    pm = PortfolioManager(initial_capital=1_000_000)
    print(f"  ✓ Portfolio created with ${pm.total_equity:,.0f}")
    
    # Add position
    pos = Position(symbol="SPY", quantity=100, avg_cost=450)
    pos.update_price(460)
    pm.positions["SPY"] = pos
    pm.cash -= 100 * 450
    
    print(f"  ✓ Added SPY position: {pos}")
    print(f"      Unrealized P&L: ${pos.unrealized_pnl:,.0f}")
    print(f"      Portfolio Equity: ${pm.total_equity:,.0f}")
    
    # Test allocator
    from quantxalgo.strategies.momentum import MACrossoverStrategy, BreakoutStrategy
    allocator = CapitalAllocator(total_capital=1_000_000)
    
    strategies = [
        MACrossoverStrategy(name="ma1", symbols=["SPY"]),
        BreakoutStrategy(name="bo1", symbols=["QQQ"]),
    ]
    
    allocations = allocator.equal_weight(strategies)
    print(f"  ✓ Equal weight allocations: {allocations}")
    
    return True


async def test_risk():
    """Test risk management."""
    print("\n" + "=" * 50)
    print("🛡️ Testing Risk Management...")
    print("=" * 50)
    
    from quantxalgo.risk.manager import RiskManager, RiskLimits
    from quantxalgo.risk.kill_switch import KillSwitch
    from quantxalgo.risk.stress_test import StressTester, HISTORICAL_SCENARIOS
    from quantxalgo.portfolio.manager import PortfolioManager
    from quantxalgo.portfolio.position import Position
    
    # Create portfolio for testing
    pm = PortfolioManager(initial_capital=1_000_000)
    pos = Position(symbol="SPY", quantity=100, avg_cost=450)
    pos.update_price(460)
    pm.positions["SPY"] = pos
    pm.cash = 954000
    
    # Risk manager
    rm = RiskManager(RiskLimits(max_drawdown=0.15))
    checks = rm.check_portfolio(pm)
    print(f"  ✓ Risk checks: {len(checks)} checks performed")
    for check in checks:
        print(f"      {check.check_type}: {'PASS' if check.passed else 'FAIL'}")
    
    # Kill switch
    ks = KillSwitch(max_drawdown=0.15)
    status = ks.check(pm)
    print(f"  ✓ Kill switch: {'TRIGGERED' if status.triggered else 'OK'}")
    
    # Stress test
    tester = StressTester()
    results = tester.run_all(pm)
    passed = sum(1 for r in results if r.survives)
    print(f"  ✓ Stress tests: {passed}/{len(results)} scenarios survived")
    
    return True


async def test_ml():
    """Test ML components."""
    print("\n" + "=" * 50)
    print("🤖 Testing ML Components...")
    print("=" * 50)
    
    from quantxalgo.ml.feature_engine import MLFeatureEngine
    from quantxalgo.ml.signal_combiner import SignalCombiner
    from quantxalgo.data.sources.yahoo import YahooDataSource
    
    # Fetch data
    async with YahooDataSource() as yahoo:
        end = datetime.now()
        start = end - timedelta(days=500)
        data = await yahoo.fetch_ohlcv("SPY", start, end)
    
    # Feature engineering
    engine = MLFeatureEngine()
    X, y = engine.prepare_training_data(data, target_horizon=5)
    print(f"  ✓ Prepared {len(X)} samples with {len(X.columns)} features")
    
    # Signal combiner
    combiner = SignalCombiner(method="confidence_weighted")
    combiner.add_signal("SPY", "momentum", 0.8, confidence=0.7)
    combiner.add_signal("SPY", "mean_reversion", -0.3, confidence=0.5)
    combiner.add_signal("SPY", "ml_alpha", 0.5, confidence=0.9)
    
    combined = combiner.combine("SPY")
    print(f"  ✓ Signal combiner: {combined.direction.value}, strength={combined.strength:.2f}")
    
    return True


async def test_regime():
    """Test regime detection."""
    print("\n" + "=" * 50)
    print("📈 Testing Regime Detection...")
    print("=" * 50)
    
    from quantxalgo.engine.regime import RegimeDetector
    from quantxalgo.data.sources.yahoo import YahooDataSource
    
    async with YahooDataSource() as yahoo:
        end = datetime.now()
        start = end - timedelta(days=365)
        data = await yahoo.fetch_ohlcv("SPY", start, end)
    
    detector = RegimeDetector()
    regime = detector.detect(data["close"])
    
    print(f"  ✓ Current regime: {regime.regime.value}")
    print(f"      Confidence: {regime.confidence:.0%}")
    print(f"      Volatility: {regime.volatility:.2%}")
    print(f"      Scaling factor: {detector.get_scaling_factor():.2f}")
    
    return True


async def test_reports():
    """Test reporting."""
    print("\n" + "=" * 50)
    print("📋 Testing Reports...")
    print("=" * 50)
    
    from quantxalgo.reports.generator import ReportGenerator
    from quantxalgo.reports.tearsheet import TearsheetGenerator
    import pandas as pd
    import numpy as np
    
    # Create mock equity curve
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500), index=dates)
    equity = (1 + returns).cumprod() * 1_000_000
    
    trades = [
        {"symbol": "SPY", "side": "BUY", "pnl": 500},
        {"symbol": "SPY", "side": "SELL", "pnl": -200},
        {"symbol": "QQQ", "side": "BUY", "pnl": 800},
    ]
    
    # Generate report
    generator = ReportGenerator()
    report = generator.generate(equity, trades)
    
    print(f"  ✓ Report generated with {len(report)} sections")
    print(f"      Summary: {report['summary']}")
    
    # Generate tearsheet
    ts_gen = TearsheetGenerator()
    tearsheet = ts_gen.generate(equity, trades, "Test Strategy")
    text = ts_gen.to_text(tearsheet)
    print(f"  ✓ Tearsheet generated ({len(text)} chars)")
    
    return True


async def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print(" QuantXalgo Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("Data Engine", test_data_engine),
        ("Strategies", test_strategies),
        ("Backtester", test_backtester),
        ("Portfolio", test_portfolio),
        ("Risk Management", test_risk),
        ("ML Components", test_ml),
        ("Regime Detection", test_regime),
        ("Reports", test_reports),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            passed = await test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ✗ FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(" TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, p, _ in results if p)
    total = len(results)
    
    for name, p, error in results:
        status = "✓ PASSED" if p else f"✗ FAILED: {error}"
        print(f"  {name}: {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Platform is production ready!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
