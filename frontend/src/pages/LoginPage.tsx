import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import './LoginPage.css'

export default function LoginPage() {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const { login, register } = useAuth()
  const navigate = useNavigate()

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError('')

    if (!username.trim() || !password) {
      setError('请输入用户名和密码')
      return
    }

    if (mode === 'register' && password !== confirmPassword) {
      setError('两次密码输入不一致')
      return
    }

    setLoading(true)
    try {
      if (mode === 'login') {
        await login(username.trim(), password)
      } else {
        await register(username.trim(), password)
      }
      navigate('/qa', { replace: true })
    } catch (err: any) {
      const detail = err?.response?.data?.detail
      if (detail) {
        setError(typeof detail === 'string' ? detail : JSON.stringify(detail))
      } else if (mode === 'login') {
        setError('登录失败，请检查用户名和密码')
      } else {
        setError('注册失败，请重试')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="login-page">
      <div className="login-card">
        <h2>KnowRAG</h2>
        <p className="subtitle">个人知识库管理系统</p>

        <div className="tabs">
          <button
            className={`tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => { setMode('login'); setError('') }}
          >
            登录
          </button>
          <button
            className={`tab ${mode === 'register' ? 'active' : ''}`}
            onClick={() => { setMode('register'); setError('') }}
          >
            注册
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="field">
            <label>用户名</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="输入用户名"
              autoComplete="username"
            />
          </div>

          <div className="field">
            <label>密码</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="输入密码"
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
            />
          </div>

          {mode === 'register' && (
            <div className="field">
              <label>确认密码</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="再次输入密码"
                autoComplete="new-password"
              />
            </div>
          )}

          <div className="error">{error || ' '}</div>

          <button className="submit-btn" type="submit" disabled={loading}>
            {loading ? '请稍候...' : mode === 'login' ? '登录' : '注册'}
          </button>
        </form>
      </div>
    </div>
  )
}
