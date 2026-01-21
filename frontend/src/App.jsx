import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import QuantLab from './pages/QuantLab'
import Portfolio from './pages/Portfolio'
import Reports from './pages/Reports'
import RiskMgmt from './pages/RiskMgmt'
import './App.css'

function App() {
  return (
    <Router>
      <div className="app">
        <aside className="sidebar">
          <div className="logo">
            <span className="logo-icon">⚡</span>
            <div className="logo-text">
              <span className="logo-name">QUANTXALGO</span>
              <span className="logo-sub">Hedge Fund</span>
            </div>
          </div>
          
          <nav className="nav">
            <NavLink to="/" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">📊</span>
              Dashboard
            </NavLink>
            <NavLink to="/quant-lab" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">🧪</span>
              Quant Lab
            </NavLink>
            <NavLink to="/portfolio" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">💼</span>
              Portfolio
            </NavLink>
            <NavLink to="/reports" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">📈</span>
              Reports
            </NavLink>
            <NavLink to="/risk" className={({isActive}) => `nav-item ${isActive ? 'active' : ''}`}>
              <span className="nav-icon">🛡️</span>
              Risk Mgmt
            </NavLink>
          </nav>
          
          <div className="sidebar-footer">
            <div className="status-indicator">
              <span className="status-dot online"></span>
              API Connected
            </div>
          </div>
        </aside>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/quant-lab" element={<QuantLab />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/reports" element={<Reports />} />
            <Route path="/risk" element={<RiskMgmt />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
