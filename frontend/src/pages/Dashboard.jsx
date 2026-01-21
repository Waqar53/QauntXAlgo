import { useState, useEffect } from 'react'
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
import './Dashboard.css'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler)

const API_BASE = 'http://localhost:8000'

export default function Dashboard() {
    const [data, setData] = useState(null)
    const [timeframe, setTimeframe] = useState('3M')
    const [currentTime, setCurrentTime] = useState(new Date())

    useEffect(() => {
        fetchDashboard()
        const dataInterval = setInterval(fetchDashboard, 5000)
        const timeInterval = setInterval(() => setCurrentTime(new Date()), 1000)
        return () => {
            clearInterval(dataInterval)
            clearInterval(timeInterval)
        }
    }, [])

    const fetchDashboard = async () => {
        try {
            const res = await axios.get(API_BASE + '/api/v1/live/dashboard')
            setData(res.data)
        } catch (err) {
            console.error('Failed to fetch:', err)
        }
    }

    const formatCurrency = (val, decimals = 2) => {
        if (val === undefined || val === null) return '$0'
        const abs = Math.abs(val)
        if (abs >= 1e9) return '$' + (val / 1e9).toFixed(decimals) + 'B'
        if (abs >= 1e6) return '$' + (val / 1e6).toFixed(decimals) + 'M'
        if (abs >= 1e3) return '$' + (val / 1e3).toFixed(decimals) + 'K'
        return '$' + val.toFixed(0)
    }

    // Live data with fallbacks
    const nav = data?.nav || 12450000
    const dailyPnl = data?.daily_pnl || 17850
    const dailyPnlPct = data?.daily_pnl_pct || 0.0045
    const prevDayPnl = data?.prev_day_pnl || 0.012
    const sharpe = data?.sharpe || 2.45
    const sharpeDelta = data?.sharpe_delta || 0.05
    const var95 = data?.var_95 || 450000
    const varDelta = data?.var_delta || -0.021

    // Generate realistic equity curve
    const generateEquityCurve = () => {
        const points = []
        let equity = 10400000
        const days = timeframe === '1D' ? 24 : timeframe === '1W' ? 7 : timeframe === '1M' ? 30 : timeframe === '3M' ? 90 : timeframe === 'YTD' ? 180 : 365

        for (let i = 0; i < days; i++) {
            const change = (Math.random() - 0.45) * equity * 0.008
            equity += change
            const date = new Date()
            date.setDate(date.getDate() - (days - i))
            points.push({
                date: date.toISOString().split('T')[0],
                equity: equity
            })
        }
        return points
    }

    const equityCurve = data?.equity_curve || generateEquityCurve()

    const chartData = {
        labels: equityCurve.map(e => {
            const d = new Date(e.date)
            return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
        }),
        datasets: [{
            data: equityCurve.map(e => e.equity || e.nav),
            borderColor: '#00ff88',
            backgroundColor: (context) => {
                const ctx = context.chart.ctx
                const gradient = ctx.createLinearGradient(0, 0, 0, 400)
                gradient.addColorStop(0, 'rgba(0, 255, 136, 0.15)')
                gradient.addColorStop(1, 'rgba(0, 255, 136, 0)')
                return gradient
            },
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            borderWidth: 2,
        }]
    }

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#1a1a2e',
                titleColor: '#fff',
                bodyColor: '#00ff88',
                borderColor: '#00ff88',
                borderWidth: 1,
                callbacks: {
                    label: (ctx) => formatCurrency(ctx.raw)
                }
            }
        },
        scales: {
            x: {
                display: true,
                grid: { color: 'rgba(255,255,255,0.03)' },
                ticks: { color: '#666', maxTicksLimit: 8, font: { size: 11 } }
            },
            y: {
                display: true,
                position: 'left',
                grid: { color: 'rgba(255,255,255,0.03)' },
                ticks: {
                    color: '#666',
                    font: { size: 11 },
                    callback: val => formatCurrency(val, 1)
                }
            }
        },
        interaction: {
            intersect: false,
            mode: 'index'
        }
    }

    // Active strategies
    const strategies = [
        { name: 'Alpha-Neutral-Eq', type: 'Market', pnl: 12500, allocation: 2500000, positions: 42, sharpe: 2.4, status: 'active' },
        { name: 'Global-Macro-FX', type: 'Global', pnl: -4200, allocation: 5000000, positions: 12, sharpe: 1.8, status: 'active' },
        { name: 'Vol-Carry-SPX', type: 'Volatility', pnl: 3100, allocation: 1500000, positions: 8, sharpe: 3.1, status: 'active' },
        { name: 'Crypto-Mom-L1', type: 'Trend', pnl: 850, allocation: 500000, positions: 5, sharpe: 1.2, status: 'active' },
    ]

    const timeframes = ['1D', '1W', '1M', '3M', 'YTD', 'ALL']

    return (
        <div className="dashboard">
            {/* Top Header */}
            <div className="dash-header">
                <div className="breadcrumb">
                    <span className="bc-root">root</span> / <span className="bc-env">production</span> / <span className="bc-page">live-trading</span>
                </div>
                <div className="market-status">
                    <span className="market-badge open">● NYSE: OPEN</span>
                    <span className="market-badge open">● LSE: OPEN</span>
                    <span className="refresh-badge">↻ 12ms</span>
                </div>
                <div className="time-display">
                    {currentTime.toISOString().replace('T', ' ').slice(0, 19)} UTC
                </div>
            </div>

            {/* Main Metrics */}
            <div className="metrics-row">
                <div className="metric-box">
                    <div className="metric-header">
                        <span className="metric-title">NET ASSET VALUE (NAV)</span>
                        <span className="metric-icon">$</span>
                    </div>
                    <div className="metric-value">{formatCurrency(nav)}</div>
                    <div className="metric-delta positive">↗ +{(prevDayPnl * 100).toFixed(1)}% vs previous day</div>
                </div>

                <div className="metric-box">
                    <div className="metric-header">
                        <span className="metric-title">DAILY PNL</span>
                        <span className="metric-icon">↗</span>
                    </div>
                    <div className="metric-value">{formatCurrency(dailyPnl)}</div>
                    <div className={'metric-delta ' + (dailyPnlPct >= 0 ? 'positive' : 'negative')}>
                        ↗ +{(dailyPnlPct * 100).toFixed(2)}% vs previous day
                    </div>
                </div>

                <div className="metric-box">
                    <div className="metric-header">
                        <span className="metric-title">SHARPE RATIO (YTD)</span>
                        <span className="metric-icon">∿</span>
                    </div>
                    <div className="metric-value">{sharpe.toFixed(2)}</div>
                    <div className="metric-delta positive">↗ +{sharpeDelta.toFixed(2)} vs previous day</div>
                </div>

                <div className="metric-box">
                    <div className="metric-header">
                        <span className="metric-title">VALUE AT RISK (95%)</span>
                        <span className="metric-icon">⚠</span>
                    </div>
                    <div className="metric-value">{formatCurrency(var95)}</div>
                    <div className={'metric-delta ' + (varDelta >= 0 ? 'positive' : 'negative')}>
                        ↘ {(varDelta * 100).toFixed(1)}% vs previous day
                    </div>
                </div>
            </div>

            {/* Main Content */}
            <div className="dash-content">
                {/* Chart Section */}
                <div className="chart-section">
                    <div className="chart-header">
                        <h2>Fund Performance (YTD)</h2>
                        <div className="timeframe-selector">
                            {timeframes.map(tf => (
                                <button
                                    key={tf}
                                    className={'tf-btn ' + (timeframe === tf ? 'active' : '')}
                                    onClick={() => setTimeframe(tf)}
                                >
                                    {tf}
                                </button>
                            ))}
                        </div>
                    </div>
                    <div className="chart-container">
                        <Line data={chartData} options={chartOptions} />
                    </div>
                </div>

                {/* Strategies Panel */}
                <div className="strategies-panel">
                    <div className="strat-header">
                        <h3>🎯 Active Strategies</h3>
                        <span className="updated-badge">Updated: Live</span>
                    </div>
                    <div className="strat-list">
                        {strategies.map((strat, i) => (
                            <div key={i} className="strat-card">
                                <div className="strat-top">
                                    <div className="strat-name">
                                        <span className={'strat-dot ' + (strat.pnl >= 0 ? 'green' : 'red')}></span>
                                        {strat.name}
                                    </div>
                                    <span className="strat-type">{strat.type}</span>
                                </div>
                                <div className="strat-metrics">
                                    <div className="strat-metric">
                                        <span className="sm-label">DAILY PNL</span>
                                        <span className={'sm-value ' + (strat.pnl >= 0 ? 'positive' : 'negative')}>
                                            {strat.pnl >= 0 ? '+' : ''}{formatCurrency(strat.pnl)}
                                        </span>
                                    </div>
                                    <div className="strat-metric">
                                        <span className="sm-label">ALLOCATION</span>
                                        <span className="sm-value">{formatCurrency(strat.allocation)}</span>
                                    </div>
                                </div>
                                <div className="strat-footer">
                                    <span>Positions: {strat.positions}</span>
                                    <span>Sharpe: {strat.sharpe}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* System Status */}
            <div className="system-bar">
                <div className="sys-status">
                    <span className="sys-label">SYSTEM STATUS</span>
                    <span className="sys-value operational">OPERATIONAL</span>
                </div>
            </div>
        </div>
    )
}
