import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from 'react'
import { api } from '../api/client'

interface AuthState {
  token: string | null
  username: string | null
  isAuthenticated: boolean
}

interface AuthContextType extends AuthState {
  login: (username: string, password: string) => Promise<void>
  register: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(
    () => localStorage.getItem('token')
  )
  const [username, setUsername] = useState<string | null>(
    () => localStorage.getItem('username')
  )

  const isAuthenticated = token !== null

  // Keep localStorage in sync with state
  useEffect(() => {
    if (token) {
      localStorage.setItem('token', token)
    } else {
      localStorage.removeItem('token')
    }
  }, [token])

  useEffect(() => {
    if (username) {
      localStorage.setItem('username', username)
    } else {
      localStorage.removeItem('username')
    }
  }, [username])

  const login = useCallback(async (uname: string, password: string) => {
    const resp = await api.post('/auth/login', { username: uname, password })
    const data = resp.data as { access_token: string; token_type: string }
    setToken(data.access_token)
    setUsername(uname)
  }, [])

  const register = useCallback(async (uname: string, password: string) => {
    await api.post('/auth/register', { username: uname, password })
    // Auto-login after register
    await login(uname, password)
  }, [login])

  const logout = useCallback(() => {
    setToken(null)
    setUsername(null)
  }, [])

  return (
    <AuthContext.Provider
      value={{ token, username, isAuthenticated, login, register, logout }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth(): AuthContextType {
  const ctx = useContext(AuthContext)
  if (!ctx) {
    throw new Error('useAuth must be used within AuthProvider')
  }
  return ctx
}

export default AuthContext
