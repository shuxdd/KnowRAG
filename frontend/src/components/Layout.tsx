import { NavLink, Outlet } from 'react-router-dom'

const sidebarStyle: React.CSSProperties = {
  width: 220,
  background: 'var(--sidebar-bg)',
  color: 'var(--sidebar-text)',
  padding: '24px 0',
  display: 'flex',
  flexDirection: 'column',
  position: 'fixed',
  top: 0,
  left: 0,
  bottom: 0,
}

const logoStyle: React.CSSProperties = {
  padding: '0 24px 32px',
  fontSize: 20,
  fontWeight: 700,
  color: '#fff',
}

const linkBase: React.CSSProperties = {
  display: 'block',
  padding: '12px 24px',
  color: 'var(--sidebar-text)',
  textDecoration: 'none',
  fontSize: 15,
  transition: 'background 0.2s',
}

const mainStyle: React.CSSProperties = {
  marginLeft: 220,
  flex: 1,
  padding: '32px 40px',
}

export default function Layout() {
  return (
    <div style={{ display: 'flex', minHeight: '100vh' }}>
      <nav style={sidebarStyle}>
        <div style={logoStyle}>KnowRAG</div>
        <NavLink
          to="/qa"
          style={({ isActive }) => ({
            ...linkBase,
            background: isActive ? 'rgba(255,255,255,0.1)' : 'transparent',
            color: isActive ? '#fff' : 'var(--sidebar-text)',
          })}
        >
          智能问答
        </NavLink>
        <NavLink
          to="/documents"
          style={({ isActive }) => ({
            ...linkBase,
            background: isActive ? 'rgba(255,255,255,0.1)' : 'transparent',
            color: isActive ? '#fff' : 'var(--sidebar-text)',
          })}
        >
          文档管理
        </NavLink>
        <NavLink
          to="/eval"
          style={({ isActive }) => ({
            ...linkBase,
            background: isActive ? 'rgba(255,255,255,0.1)' : 'transparent',
            color: isActive ? '#fff' : 'var(--sidebar-text)',
          })}
        >
          评估报告
        </NavLink>
      </nav>
      <main style={mainStyle}><Outlet /></main>
    </div>
  )
}
