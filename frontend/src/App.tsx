import { Routes, Route, Navigate, Outlet } from 'react-router-dom'
import Layout from './components/Layout'
import DocumentsPage from './pages/DocumentsPage'
import QAPage from './pages/QAPage'
import EvalPage from './pages/EvalPage'
import LoginPage from './pages/LoginPage'
import { useAuth } from './contexts/AuthContext'
import './App.css'

function ProtectedRoute() {
  const { isAuthenticated } = useAuth()
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />
  }
  return <Outlet />
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route element={<ProtectedRoute />}>
        <Route element={<Layout />}>
          <Route path="/qa" element={<QAPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/eval" element={<EvalPage />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/qa" replace />} />
    </Routes>
  )
}

export default App
