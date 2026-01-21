import { useState, useRef, useEffect } from 'react'
import Editor from '@monaco-editor/react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import './QuantLab.css'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const API_BASE = 'http://localhost:8000'

const DEFAULT_CODE = `class MyStrategy(QuantLabStrategy):
    """
    Write your trading algorithm here.
    
    Available methods:
    - initialize(): Called once at start
    - on_bar(data, context): Called on each bar, return list of Orders
    - market_order(symbol, qty): Create market order
    - close_position(symbol): Close existing position
    - log(message): Log to console
    """
    
    def initialize(self):
        self.lookback = 20
        self.position_pct = 0.10
        self.log("Strategy initialized!")
        
    def on_bar(self, data, context):
        orders = []
        
        for symbol in self.symbols:
            if symbol not in data:
                continue
            df = data[symbol]
            if len(df) < self.lookback:
                continue
                
            close = df["close"]
            price = close.iloc[-1]
            sma = close.rolling(self.lookback).mean().iloc[-1]
            
            # Buy when price crosses above SMA
            if price > sma and not context.has_position(symbol):
                qty = int(context.equity * self.position_pct / price)
                self.log(f"BUY {symbol}: Price {price:.2f} > SMA {sma:.2f}")
                orders.append(self.market_order(symbol, qty))
                
            # Sell when price crosses below SMA
            elif price < sma and context.has_position(symbol):
                self.log(f"SELL {symbol}: Price {price:.2f} < SMA {sma:.2f}")
                orders.append(self.close_position(symbol))
                
        return orders
`

const STRATEGY_FILES = [
  { name: 'my_strategy.py', icon: '📄', active: true },
  { name: 'momentum.py', icon: '📄' },
  { name: 'mean_reversion.py', icon: '📄' },
  { name: 'pairs_trading.py', icon: '📄' },
]

const SHARED_LIBS = [
  { name: 'indicators.py', icon: '📄' },
  { name: 'risk_models.py', icon: '📄' },
  { name: 'execution.py', icon: '📄' },
]

export default function QuantLab() {
  const [code, setCode] = useState(DEFAULT_CODE)
  const [logs, setLogs] = useState([
    { type: 'system', text: 'Quantum Research Console [v2.1.0]' },
    { type: 'info', text: 'user@quantxalgo:~/strategies$ Ready to execute...' },
  ])
  const [results, setResults] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [activeFile, setActiveFile] = useState('my_strategy.py')
  const [config, setConfig] = useState({
    symbols: 'NVDA,TSLA,AAPL,MSFT,META',
    startDate: '2023-01-01',
    endDate: '2024-06-01',
    capital: '50000000'
  })

  const terminalRef = useRef(null)

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs])

  const addLog = (text, type = 'info') => {
    setLogs(prev => [...prev, { type, text, time: new Date().toLocaleTimeString() }])
  }

  const runBacktest = async () => {
    setIsRunning(true)
    setResults(null)
    setLogs([
      { type: 'system', text: 'Quantum Research Console [v2.1.0]' },
      { type: 'cmd', text: 'user@quantxalgo:~/strategies$ python run_backtest.py' },
    ])

    addLog('> Loading historical data...', 'info')
    addLog('> Parsing strategy code...', 'info')

    try {
      const res = await axios.post(API_BASE + '/api/v1/quant-lab/execute', {
        code,
        symbols: config.symbols.split(',').map(s => s.trim()),
        start_date: config.startDate,
        end_date: config.endDate,
        initial_capital: parseFloat(config.capital),
        params: {}
      })

      const data = res.data

      if (!data.success) {
        addLog('> ERROR: ' + data.error, 'error')
        setIsRunning(false)
        return
      }

      // Add execution logs
      if (data.logs) {
        data.logs.forEach(log => {
          const type = log.includes('ERROR') ? 'error' :
            log.includes('TRADE') ? 'trade' :
              log.includes('🟢') ? 'buy' :
                log.includes('🔴') ? 'sell' : 'info'
          addLog(log, type)
        })
      }

      addLog('', 'info')
      addLog('═'.repeat(60), 'success')
      addLog('  BACKTEST COMPLETE', 'success')
      addLog('═'.repeat(60), 'success')
      addLog('  Final Equity: $' + data.final_equity.toLocaleString(), 'success')
      addLog('  Total Return: ' + (data.total_return * 100).toFixed(2) + '%', 'success')
      addLog('  Sharpe Ratio: ' + data.sharpe_ratio.toFixed(2), 'success')
      addLog('  Total Trades: ' + data.total_trades, 'success')

      setResults(data)

    } catch (err) {
      addLog('> ERROR: ' + err.message, 'error')
    }

    setIsRunning(false)
  }

  const formatCurrency = (val) => {
    if (val >= 1e9) return '$' + (val / 1e9).toFixed(2) + 'B'
    if (val >= 1e6) return '$' + (val / 1e6).toFixed(2) + 'M'
    if (val >= 1e3) return '$' + (val / 1e3).toFixed(1) + 'K'
    return '$' + val.toFixed(0)
  }

  const chartData = results?.equity_curve ? {
    labels: results.equity_curve.map(e => e.date?.slice(5) || ''),
    datasets: [{
      label: 'Equity',
      data: results.equity_curve.map(e => e.equity),
      borderColor: '#00ff88',
      backgroundColor: 'rgba(0, 255, 136, 0.1)',
      fill: true,
      tension: 0.4,
      pointRadius: 0,
    }]
  } : null

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        display: true,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#666', maxTicksLimit: 10 }
      },
      y: {
        display: true,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: {
          color: '#666',
          callback: val => formatCurrency(val)
        }
      }
    }
  }

  return (
    <div className="quantlab">
      {/* Header */}
      <div className="quantlab-header">
        <div className="header-left">
          <h1>🧪 QUANT LAB</h1>
          <span className="breadcrumb">Research / Strategies / {activeFile}</span>
        </div>
        <div className="header-right">
          <button className="btn btn-secondary" disabled={isRunning}>
            ✓ Validate
          </button>
          <button
            className="btn btn-primary"
            onClick={runBacktest}
            disabled={isRunning}
          >
            {isRunning ? '⏳ Running...' : '▶ RUN BACKTEST'}
          </button>
        </div>
      </div>

      {/* Main IDE Layout */}
      <div className="ide-container">
        {/* File Explorer */}
        <div className="file-explorer">
          <div className="explorer-section">
            <div className="explorer-header">📁 WORKSPACE</div>
            {STRATEGY_FILES.map(file => (
              <div
                key={file.name}
                className={'file-item' + (activeFile === file.name ? ' active' : '')}
                onClick={() => setActiveFile(file.name)}
              >
                {file.icon} {file.name}
              </div>
            ))}
          </div>

          <div className="explorer-section">
            <div className="explorer-header">📦 SHARED LIBS</div>
            {SHARED_LIBS.map(file => (
              <div key={file.name} className="file-item">
                {file.icon} {file.name}
              </div>
            ))}
          </div>

          <div className="config-section">
            <div className="explorer-header">⚙️ CONFIG</div>
            <div className="config-item">
              <label>Symbols</label>
              <input
                value={config.symbols}
                onChange={e => setConfig({ ...config, symbols: e.target.value })}
              />
            </div>
            <div className="config-item">
              <label>Capital</label>
              <input
                value={config.capital}
                onChange={e => setConfig({ ...config, capital: e.target.value })}
              />
            </div>
            <div className="config-item">
              <label>Start</label>
              <input
                type="date"
                value={config.startDate}
                onChange={e => setConfig({ ...config, startDate: e.target.value })}
              />
            </div>
            <div className="config-item">
              <label>End</label>
              <input
                type="date"
                value={config.endDate}
                onChange={e => setConfig({ ...config, endDate: e.target.value })}
              />
            </div>
          </div>
        </div>

        {/* Editor & Terminal */}
        <div className="editor-panel">
          <div className="editor-tabs">
            <div className="tab active">{activeFile}</div>
            <span className="python-badge">Python 3.11</span>
          </div>

          <div className="editor-wrapper">
            <Editor
              height="100%"
              language="python"
              theme="vs-dark"
              value={code}
              onChange={setCode}
              options={{
                fontSize: 14,
                fontFamily: "'Fira Code', monospace",
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                lineNumbers: 'on',
                renderLineHighlight: 'line',
                automaticLayout: true,
              }}
            />
          </div>

          {/* Terminal */}
          <div className="terminal" ref={terminalRef}>
            <div className="terminal-header">
              <span>📟 TERMINAL</span>
              <span className="terminal-status">
                {isRunning ? '⏳ Executing...' : '✓ Ready'}
              </span>
            </div>
            <div className="terminal-content">
              {logs.map((log, i) => (
                <div key={i} className={'log-line ' + log.type}>
                  {log.text}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Results Panel */}
        <div className="results-panel">
          <div className="results-header">📊 RESULTS</div>

          {results ? (
            <>
              {/* P&L Hero */}
              <div className={'pnl-hero ' + (results.total_return >= 0 ? 'profit' : 'loss')}>
                <div className="pnl-label">TOTAL P&L</div>
                <div className="pnl-value">
                  {results.total_return >= 0 ? '+' : ''}
                  {formatCurrency(results.final_equity - results.initial_capital)}
                </div>
                <div className="pnl-percent">
                  {results.total_return >= 0 ? '+' : ''}
                  {(results.total_return * 100).toFixed(2)}%
                </div>
              </div>

              {/* Equity Chart */}
              {chartData && (
                <div className="chart-container">
                  <Line data={chartData} options={chartOptions} />
                </div>
              )}

              {/* Metrics Grid */}
              <div className="results-metrics">
                <div className="result-metric">
                  <span className="metric-name">Sharpe</span>
                  <span className="metric-val">{results.sharpe_ratio.toFixed(2)}</span>
                </div>
                <div className="result-metric">
                  <span className="metric-name">Sortino</span>
                  <span className="metric-val">{results.sortino_ratio.toFixed(2)}</span>
                </div>
                <div className="result-metric">
                  <span className="metric-name">Max DD</span>
                  <span className="metric-val negative">{(results.max_drawdown * 100).toFixed(1)}%</span>
                </div>
                <div className="result-metric">
                  <span className="metric-name">Calmar</span>
                  <span className="metric-val">{results.calmar_ratio.toFixed(2)}</span>
                </div>
                <div className="result-metric">
                  <span className="metric-name">Trades</span>
                  <span className="metric-val">{results.total_trades}</span>
                </div>
                <div className="result-metric">
                  <span className="metric-name">Win Rate</span>
                  <span className="metric-val">{(results.win_rate * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* Recent Trades */}
              {results.trades && results.trades.length > 0 && (
                <div className="trades-section">
                  <div className="trades-header">Recent Trades</div>
                  <div className="trades-list">
                    {results.trades.slice(-10).map((t, i) => (
                      <div key={i} className={'trade-item ' + t.side}>
                        <span className="trade-side">{t.side === 'BUY' ? '🟢' : '🔴'}</span>
                        <span className="trade-symbol">{t.symbol}</span>
                        <span className="trade-qty">{t.quantity}</span>
                        <span className="trade-price">${t.price?.toFixed(2)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="no-results">
              <p>Run a backtest to see results</p>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
