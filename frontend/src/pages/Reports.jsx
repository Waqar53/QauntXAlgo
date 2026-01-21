export default function Reports() {
    const trades = [
        { date: '2024-01-15', symbol: 'NVDA', side: 'BUY', qty: 500, price: 450.25, value: 225125, pnl: 0 },
        { date: '2024-01-14', symbol: 'AAPL', side: 'SELL', qty: 200, price: 185.50, value: 37100, pnl: 2340 },
        { date: '2024-01-14', symbol: 'MSFT', side: 'BUY', qty: 300, price: 378.20, value: 113460, pnl: 0 },
        { date: '2024-01-13', symbol: 'TSLA', side: 'SELL', qty: 150, price: 242.80, value: 36420, pnl: -890 },
        { date: '2024-01-12', symbol: 'META', side: 'BUY', qty: 400, price: 485.00, value: 194000, pnl: 0 },
        { date: '2024-01-11', symbol: 'GOOGL', side: 'SELL', qty: 600, price: 142.30, value: 85380, pnl: 4560 },
        { date: '2024-01-10', symbol: 'AMZN', side: 'BUY', qty: 800, price: 156.75, value: 125400, pnl: 0 },
    ]

    return (
        <div className="page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">📈 Reports</h1>
                    <p className="page-subtitle">Performance Analytics & Trade History</p>
                </div>
            </div>

            <div className="metrics-grid" style={{ marginBottom: '24px' }}>
                <div className="metric-card">
                    <div className="metric-label">YTD Return</div>
                    <div className="metric-value positive">+18.5%</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">MTD Return</div>
                    <div className="metric-value positive">+3.2%</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Total Trades</div>
                    <div className="metric-value">1,247</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Win Rate</div>
                    <div className="metric-value">58.3%</div>
                </div>
            </div>

            <div className="card">
                <div className="card-header">Trade History</div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Symbol</th>
                            <th>Side</th>
                            <th>Qty</th>
                            <th>Price</th>
                            <th>Value</th>
                            <th>P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trades.map((t, i) => (
                            <tr key={i}>
                                <td>{t.date}</td>
                                <td><strong>{t.symbol}</strong></td>
                                <td style={{ color: t.side === 'BUY' ? '#00ff88' : '#ff4c4c' }}>{t.side}</td>
                                <td>{t.qty.toLocaleString()}</td>
                                <td>${t.price.toFixed(2)}</td>
                                <td>${(t.value / 1e3).toFixed(1)}K</td>
                                <td style={{ color: t.pnl >= 0 ? '#00ff88' : '#ff4c4c' }}>
                                    {t.pnl !== 0 ? `${t.pnl >= 0 ? '+' : ''}$${t.pnl.toLocaleString()}` : '-'}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    )
}
