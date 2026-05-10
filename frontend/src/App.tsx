import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import DocumentsPage from './pages/DocumentsPage'
import QAPage from './pages/QAPage'
import EvalPage from './pages/EvalPage'
import './App.css'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/documents" element={<DocumentsPage />} />
        <Route path="/qa" element={<QAPage />} />
        <Route path="/eval" element={<EvalPage />} />
        <Route path="*" element={<Navigate to="/qa" replace />} />
      </Routes>
    </Layout>
  )
}

export default App
