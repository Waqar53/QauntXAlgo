export default function Portfolio() {
    const positions = [
        { symbol: 'NVDA', name: 'NVIDIA Corp', qty: 500, avgCost: 450.25, current: 475.80, value: 237900, pnl: 12775, pnlPct: 5.67 },
        { symbol: 'AAPL', name: 'Apple Inc', qty: 1000, avgCost: 182.50, current: 178.20, value: 178200, pnl: -4300, pnlPct: -2.35 },
        { symbol: 'MSFT', name: 'Microsoft', qty: 750, avgCost: 380.00, current: 392.45, value: 294337, pnl: 9337, pnlPct: 3.27 },
        { symbol: 'TSLA', name: 'Tesla Inc', qty: 300, avgCost: 245.00, current: 238.90, value: 71670, pnl: -1830, pnlPct: -2.49 },
        { symbol: 'META', name: 'Meta Platforms', qty: 400, avgCost: 485.00, current: 512.30, value: 204920, pnl: 10920, pnlPct: 5.63 },
    ]

    const totalValue = positions.reduce((sum, p) => sum + p.value, 0)
    const totalPnl = positions.reduce((sum, p) => sum + p.pnl, 0)

    return (
        <div className="page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">💼 Portfolio</h1>
                    <p className="page-subtitle">Position Management & Allocation</p>
                </div>
            </div>

            <div className="metrics-grid" style={{ marginBottom: '24px' }}>
                <div className="metric-card">
                    <div className="metric-label">Total Value</div>
                    <div className="metric-value">${(totalValue / 1e6).toFixed(2)}M</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Unrealized P&L</div>
                    <div className={`metric-value ${totalPnl >= 0 ? 'positive' : 'negative'}`}>
                        {totalPnl >= 0 ? '+' : ''}${(totalPnl / 1e3).toFixed(1)}K
                    </div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Positions</div>
                    <div className="metric-value">{positions.length}</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Cash</div>
                    <div className="metric-value">$1.2M</div>
                </div>
            </div>

            <div className="card">
                <div className="card-header">Holdings</div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th>Name</th>
                            <th>Quantity</th>
                            <th>Avg Cost</th>
                            <th>Current</th>
                            <th>Value</th>
                            <th>P&L</th>
                            <th>%</th>
                        </tr>
                    </thead>
                    <tbody>
                        {positions.map((p, i) => (
                            <tr key={i}>
                                <td><strong>{p.symbol}</strong></td>
                                <td style={{ color: '#666' }}>{p.name}</td>
                                <td>{p.qty.toLocaleString()}</td>
                                <td>${p.avgCost.toFixed(2)}</td>
                                <td>${p.current.toFixed(2)}</td>
                                <td>${(p.value / 1e3).toFixed(1)}K</td>
                                <td style={{ color: p.pnl >= 0 ? '#00ff88' : '#ff4c4c' }}>
                                    {p.pnl >= 0 ? '+' : ''}${(p.pnl / 1e3).toFixed(1)}K
                                </td>
                                <td style={{ color: p.pnlPct >= 0 ? '#00ff88' : '#ff4c4c' }}>
                                    {p.pnlPct >= 0 ? '+' : ''}{p.pnlPct.toFixed(2)}%
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
