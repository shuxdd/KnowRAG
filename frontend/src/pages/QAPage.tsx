import { useState, useEffect, useRef, useCallback } from 'react'
import {
  askQuestionStream,
  listSessions,
  deleteSession,
  getSession,
  Source,
  SessionInfo,
  MessageInfo,
} from '../api/client'

const pageTitle: React.CSSProperties = {
  fontSize: 24, fontWeight: 600, marginBottom: 24,
}

const layoutStyle: React.CSSProperties = {
  display: 'flex', gap: 24, height: 'calc(100vh - 120px)',
}

const sidebarStyle: React.CSSProperties = {
  width: 240, flexShrink: 0, background: 'var(--card-bg)',
  borderRadius: 12, padding: 16, overflowY: 'auto',
}

const sessionItem: React.CSSProperties = {
  padding: '10px 12px', borderRadius: 8, cursor: 'pointer',
  fontSize: 14, marginBottom: 4, display: 'flex',
  justifyContent: 'space-between', alignItems: 'center',
}

const newChatBtn: React.CSSProperties = {
  width: '100%', padding: '10px', background: 'var(--primary)',
  color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer',
  fontSize: 14, marginBottom: 12,
}

const chatArea: React.CSSProperties = {
  flex: 1, display: 'flex', flexDirection: 'column',
  background: 'var(--card-bg)', borderRadius: 12, overflow: 'hidden',
}

const messagesContainer: React.CSSProperties = {
  flex: 1, overflowY: 'auto', padding: 20,
}

const msgRow: React.CSSProperties = {
  marginBottom: 16, display: 'flex', flexDirection: 'column',
}

const userBubble: React.CSSProperties = {
  alignSelf: 'flex-end', background: 'var(--primary)',
  color: '#fff', padding: '10px 16px', borderRadius: 12,
  maxWidth: '70%', fontSize: 14, lineHeight: 1.6,
}

const aiBubble: React.CSSProperties = {
  alignSelf: 'flex-start', background: '#f1f5f9',
  color: '#1a1a2e', padding: '10px 16px', borderRadius: 12,
  maxWidth: '85%', fontSize: 14, lineHeight: 1.8, whiteSpace: 'pre-wrap',
}

const inputArea: React.CSSProperties = {
  display: 'flex', gap: 8, padding: '16px 20px', borderTop: '1px solid var(--border)',
}

const inputStyle: React.CSSProperties = {
  flex: 1, padding: '10px 14px', borderRadius: 8,
  border: '1px solid var(--border)', fontSize: 14, outline: 'none',
}

const selectStyle: React.CSSProperties = {
  padding: '10px 14px', borderRadius: 8,
  border: '1px solid var(--border)', fontSize: 13,
  background: 'var(--card-bg)', cursor: 'pointer',
}

const btnStyle: React.CSSProperties = {
  padding: '10px 20px', background: 'var(--primary)', color: '#fff',
  border: 'none', borderRadius: 8, cursor: 'pointer', fontSize: 14,
}

const deleteBtn: React.CSSProperties = {
  background: 'none', border: 'none', color: '#ef4444',
  cursor: 'pointer', fontSize: 12, padding: '2px 6px',
}

const sourceChip: React.CSSProperties = {
  display: 'inline-block', background: '#e0e7ff', color: '#4338ca',
  padding: '2px 8px', borderRadius: 4, fontSize: 11, marginRight: 4, marginTop: 8,
}

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[] | null
}

export default function QAPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [strategy, setStrategy] = useState('hybrid_rerank')
  const [streaming, setStreaming] = useState(false)
  const [streamingText, setStreamingText] = useState('')
  const [streamingSources, setStreamingSources] = useState<Source[]>([])

  const msgEndRef = useRef<HTMLDivElement>(null)

  const loadSessions = useCallback(async () => {
    try {
      const res = await listSessions()
      setSessions(res.sessions)
    } catch (err) {
      console.error('Failed to load sessions:', err)
    }
  }, [])

  useEffect(() => {
    loadSessions()
  }, [loadSessions])

  useEffect(() => {
    msgEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingText])

  const handleSelectSession = async (id: string) => {
    setActiveSessionId(id)
    try {
      const res = await getSession(id)
      setMessages(res.messages.map((m: MessageInfo) => ({
        role: m.role as 'user' | 'assistant',
        content: m.content,
        sources: m.sources,
      })))
    } catch (err) {
      console.error('Failed to load session:', err)
    }
  }

  const handleNewChat = () => {
    setActiveSessionId(null)
    setMessages([])
    setStreamingText('')
  }

  const handleDeleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('Delete this conversation?')) return
    try {
      await deleteSession(id)
      if (activeSessionId === id) handleNewChat()
      await loadSessions()
    } catch (err) {
      console.error('Failed to delete:', err)
    }
  }

  const handleSend = async () => {
    if (!input.trim() || streaming) return
    const question = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setStreamingText('')
    setStreamingSources([])
    setStreaming(true)

    await askQuestionStream(
      question,
      activeSessionId,
      strategy,
      5,
      (token) => setStreamingText(prev => prev + token),
      (sources) => setStreamingSources(sources),
      (newSessionId) => {
        setStreamingText(prev => {
          setMessages(msgs => [...msgs, {
            role: 'assistant',
            content: prev,
            sources: streamingSources.length > 0 ? streamingSources : null,
          }])
          return ''
        })
        if (!activeSessionId) {
          setActiveSessionId(newSessionId)
        }
        setStreaming(false)
        loadSessions()
      },
      (err) => {
        console.error('Stream error:', err)
        alert('问答请求失败: ' + err.message)
        setStreaming(false)
      },
    )
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div>
      <h1 style={pageTitle}>智能问答</h1>
      <div style={layoutStyle}>
        <div style={sidebarStyle}>
          <button style={newChatBtn} onClick={handleNewChat}>+ 新对话</button>
          {sessions.map(s => (
            <div
              key={s.id}
              style={{
                ...sessionItem,
                background: s.id === activeSessionId ? '#eef2ff' : 'transparent',
                fontWeight: s.id === activeSessionId ? 600 : 400,
              }}
              onClick={() => handleSelectSession(s.id)}
            >
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                {s.title}
              </span>
              <button style={deleteBtn} onClick={(e) => handleDeleteSession(s.id, e)}>
                ✕
              </button>
            </div>
          ))}
        </div>

        <div style={chatArea}>
          <div style={messagesContainer}>
            {messages.map((msg, i) => (
              <div key={i} style={msgRow}>
                {msg.role === 'user' ? (
                  <div style={userBubble}>{msg.content}</div>
                ) : (
                  <div style={aiBubble}>
                    {msg.content}
                    {msg.sources && msg.sources.length > 0 && (
                      <div style={{ marginTop: 8 }}>
                        {msg.sources.map((s, j) => (
                          <span key={j} style={sourceChip}>
                            {s.filename} ({(s.score * 100).toFixed(0)}%)
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
            {streaming && (
              <div style={msgRow}>
                <div style={aiBubble}>
                  {streamingText || '思考中...'}
                  {streamingSources.length > 0 && streamingText.length > 0 && (
                    <div style={{ marginTop: 8 }}>
                      {streamingSources.map((s, j) => (
                        <span key={j} style={sourceChip}>
                          {s.filename} ({(s.score * 100).toFixed(0)}%)
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            )}
            <div ref={msgEndRef} />
          </div>

          <div style={inputArea}>
            <select style={selectStyle} value={strategy} onChange={(e) => setStrategy(e.target.value)}>
              <option value="vector">向量检索</option>
              <option value="hybrid">混合检索</option>
              <option value="hybrid_rerank">混合检索+Rerank</option>
            </select>
            <input
              style={inputStyle}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入问题，Enter 发送..."
              disabled={streaming}
            />
            <button style={btnStyle} onClick={handleSend} disabled={streaming || !input.trim()}>
              {streaming ? '...' : '发送'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
