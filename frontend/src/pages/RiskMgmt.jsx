export default function RiskMgmt() {
    const scenarios = [
        { name: '2008 Financial Crisis', impact: -32.5, probability: 'Low' },
        { name: 'COVID-19 March 2020', impact: -24.8, probability: 'Medium' },
        { name: 'Flash Crash', impact: -8.5, probability: 'Medium' },
        { name: 'Rate Shock (+200bps)', impact: -15.2, probability: 'High' },
        { name: 'Tech Sector Selloff', impact: -18.3, probability: 'Medium' },
    ]

    return (
        <div className="page">
            <div className="page-header">
                <div>
                    <h1 className="page-title">🛡️ Risk Management</h1>
                    <p className="page-subtitle">VaR Analysis & Stress Testing</p>
                </div>
                <button className="btn btn-primary" style={{ background: '#ff4c4c' }}>
                    🚨 KILL SWITCH
                </button>
            </div>

            <div className="metrics-grid" style={{ marginBottom: '24px' }}>
                <div className="metric-card">
                    <div className="metric-label">VaR (95%)</div>
                    <div className="metric-value negative">-$245K</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">VaR (99%)</div>
                    <div className="metric-value negative">-$412K</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Max Drawdown</div>
                    <div className="metric-value negative">-8.3%</div>
                </div>
                <div className="metric-card">
                    <div className="metric-label">Beta to SPY</div>
                    <div className="metric-value">1.15</div>
                </div>
            </div>

            <div className="card">
                <div className="card-header">Stress Test Scenarios</div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Scenario</th>
                            <th>Impact</th>
                            <th>P&L Impact</th>
                            <th>Probability</th>
                        </tr>
                    </thead>
                    <tbody>
                        {scenarios.map((s, i) => (
                            <tr key={i}>
                                <td><strong>{s.name}</strong></td>
                                <td style={{ color: '#ff4c4c' }}>{s.impact}%</td>
                                <td style={{ color: '#ff4c4c' }}>
                                    -${((Math.abs(s.impact) / 100) * 10000000 / 1e6).toFixed(2)}M
                                </td>
                                <td>
                                    <span style={{
                                        padding: '4px 8px',
                                        borderRadius: '4px',
                                        fontSize: '11px',
                                        background: s.probability === 'High' ? 'rgba(255,76,76,0.2)' :
                                            s.probability === 'Medium' ? 'rgba(255,165,0,0.2)' : 'rgba(0,255,136,0.2)',
                                        color: s.probability === 'High' ? '#ff4c4c' :
                                            s.probability === 'Medium' ? '#ffa500' : '#00ff88'
                                    }}>
                                        {s.probability}
                                    </span>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginTop: '24px' }}>
                <div className="card">
                    <div className="card-header">Position Limits</div>
                    <div style={{ padding: '16px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                            <span>Max Position Size</span>
                            <span style={{ color: '#00ff88' }}>15% ✓</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                            <span>Max Sector Exposure</span>
                            <span style={{ color: '#00ff88' }}>25% ✓</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                            <span>Gross Leverage</span>
                            <span style={{ color: '#00ff88' }}>1.2x ✓</span>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>Net Leverage</span>
                            <span style={{ color: '#00ff88' }}>0.85x ✓</span>
                        </div>
                    </div>
                </div>

                <div className="card">
                    <div className="card-header">Risk Alerts</div>
                    <div style={{ padding: '16px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px', padding: '12px', background: 'rgba(255,165,0,0.1)', borderRadius: '8px' }}>
                            <span>⚠️</span>
                            <div>
                                <div style={{ fontWeight: '500' }}>Tech Concentration Warning</div>
                                <div style={{ fontSize: '12px', color: '#666' }}>Technology sector at 42% (limit: 50%)</div>
                            </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', padding: '12px', background: 'rgba(0,255,136,0.1)', borderRadius: '8px' }}>
                            <span>✅</span>
                            <div>
                                <div style={{ fontWeight: '500' }}>All Limits Healthy</div>
                                <div style={{ fontSize: '12px', color: '#666' }}>No critical risk alerts</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
