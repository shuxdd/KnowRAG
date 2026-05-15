import { useState, useEffect, useRef, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import {
  askQuestionStream, askAgentStream, listSessions, deleteSession, getSession,
  Source, SessionInfo, MessageInfo,
} from '../api/client'

// ── Styles ──────────────────────────────────────────────

const page: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', height: 'calc(100vh - 88px)',
}

const header: React.CSSProperties = {
  fontSize: 22, fontWeight: 700, marginBottom: 16, color: '#1e293b',
}

const shell: React.CSSProperties = {
  display: 'flex', gap: 16, flex: 1, minHeight: 0,
}

// Sidebar
const sidebar: React.CSSProperties = {
  width: 260, flexShrink: 0, background: '#fff', borderRadius: 12,
  boxShadow: '0 1px 3px rgba(0,0,0,.06)', display: 'flex', flexDirection: 'column',
  overflow: 'hidden',
}

const sidebarHead: React.CSSProperties = {
  padding: '14px 16px', borderBottom: '1px solid #f1f5f9',
}

const newBtn: React.CSSProperties = {
  width: '100%', padding: '9px 0', background: 'var(--primary)',
  color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer',
  fontSize: 13, fontWeight: 600, letterSpacing: '.3px',
}

const sessionList: React.CSSProperties = {
  flex: 1, overflowY: 'auto', padding: '6px 8px',
}

const sessionRow = (active: boolean): React.CSSProperties => ({
  display: 'flex', alignItems: 'center', justifyContent: 'space-between',
  padding: '10px 12px', borderRadius: 8, cursor: 'pointer',
  fontSize: 13, marginBottom: 2,
  background: active ? '#eef2ff' : 'transparent',
  fontWeight: active ? 600 : 400,
  transition: 'background .15s',
})

const sessionTitle: React.CSSProperties = {
  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1,
}

const delBtn: React.CSSProperties = {
  background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer',
  fontSize: 14, padding: '0 0 0 8px', lineHeight: 1,
}

// Chat
const chat: React.CSSProperties = {
  flex: 1, display: 'flex', flexDirection: 'column',
  background: '#fff', borderRadius: 12,
  boxShadow: '0 1px 3px rgba(0,0,0,.06)', overflow: 'hidden',
}

const msgList: React.CSSProperties = {
  flex: 1, overflowY: 'auto', padding: '20px 24px',
}

const emptyState: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', alignItems: 'center',
  justifyContent: 'center', height: '100%', color: '#94a3b8',
}

// Message bubbles
const msgWrapper: React.CSSProperties = {
  marginBottom: 20, display: 'flex', flexDirection: 'column',
}

const avatar = (isUser: boolean): React.CSSProperties => ({
  width: 32, height: 32, borderRadius: 8, display: 'flex',
  alignItems: 'center', justifyContent: 'center',
  fontSize: 14, fontWeight: 700, flexShrink: 0,
  background: isUser ? 'var(--primary)' : '#f1f5f9',
  color: isUser ? '#fff' : '#64748b',
})

const bubble = (isUser: boolean): React.CSSProperties => ({
  maxWidth: '78%', padding: '12px 16px', borderRadius: 12,
  fontSize: 14, lineHeight: 1.7,
  ...(isUser
    ? { background: 'var(--primary)', color: '#fff', borderBottomRightRadius: 4 }
    : { background: '#f8fafc', color: '#1e293b', border: '1px solid #f1f5f9', borderBottomLeftRadius: 4 }
  ),
})

const markdownStyles = {
  p: { margin: '4px 0' } as React.CSSProperties,
  ul: { paddingLeft: 18, margin: '6px 0' } as React.CSSProperties,
  ol: { paddingLeft: 18, margin: '6px 0' } as React.CSSProperties,
  li: { marginBottom: 2 } as React.CSSProperties,
  strong: { fontWeight: 600 } as React.CSSProperties,
  code: { background: '#e2e8f0', padding: '1px 5px', borderRadius: 3, fontSize: 12.5 } as React.CSSProperties,
  pre: { background: '#1e293b', color: '#e2e8f0', padding: '10px 14px', borderRadius: 8, overflowX: 'auto' as const, fontSize: 12.5, margin: '8px 0' } as React.CSSProperties,
  blockquote: { borderLeft: '3px solid var(--primary)', paddingLeft: 12, color: '#64748b', margin: '8px 0' } as React.CSSProperties,
  h3: { fontSize: 15, fontWeight: 600, margin: '10px 0 4px' } as React.CSSProperties,
  h4: { fontSize: 14, fontWeight: 600, margin: '8px 0 4px' } as React.CSSProperties,
}

// Sources
const sourceRow: React.CSSProperties = { marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }

const chip = (active: boolean): React.CSSProperties => ({
  display: 'inline-flex', alignItems: 'center', gap: 4,
  padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 500,
  cursor: 'pointer', userSelect: 'none', transition: 'all .15s',
  background: active ? '#4338ca' : '#eef2ff', color: active ? '#fff' : '#4338ca',
})

const chipDot: React.CSSProperties = { width: 6, height: 6, borderRadius: 3, background: 'currentColor', opacity: .6 }

const sourcePanel: React.CSSProperties = {
  marginTop: 8, padding: '10px 14px', background: '#f8fafc',
  borderRadius: 8, fontSize: 12, color: '#475569', lineHeight: 1.7,
  whiteSpace: 'pre-wrap', maxHeight: 180, overflowY: 'auto',
  border: '1px solid #e2e8f0',
}

// Input
const inputRow: React.CSSProperties = {
  display: 'flex', gap: 8, padding: '14px 20px', borderTop: '1px solid #f1f5f9',
  alignItems: 'center',
}

const inputField: React.CSSProperties = {
  flex: 1, padding: '9px 14px', borderRadius: 8,
  border: '1px solid #e2e8f0', fontSize: 14, outline: 'none',
  background: '#f8fafc', transition: 'border-color .15s',
}

const sendBtn = (ok: boolean): React.CSSProperties => ({
  padding: '9px 22px', background: ok ? 'var(--primary)' : '#e2e8f0',
  color: ok ? '#fff' : '#94a3b8', border: 'none', borderRadius: 8,
  cursor: ok ? 'pointer' : 'default', fontSize: 13, fontWeight: 600,
  transition: 'all .15s', flexShrink: 0,
})

// ── Components ──────────────────────────────────────────

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[] | null
}

function SourcesWidget({ msgIdx, sources, expandedKey, onToggle }: {
  msgIdx: string; sources: Source[]; expandedKey: string | null; onToggle: (k: string | null) => void
}) {
  return (
    <div style={sourceRow}>
      {sources.map((s, j) => {
        const key = `${msgIdx}-${j}`
        const open = expandedKey === key
        return (
          <div key={j} style={{ width: '100%' }}>
            <span style={chip(open)} onClick={() => onToggle(open ? null : key)}>
              <span style={chipDot} />
              {s.filename}
              <span style={{ opacity: .65 }}>{(s.score * 100).toFixed(0)}%</span>
            </span>
            {open && <div style={sourcePanel}>{s.content}</div>}
          </div>
        )
      })}
    </div>
  )
}

// ── Page ────────────────────────────────────────────────

export default function QAPage() {
  const [sessions, setSessions] = useState<SessionInfo[]>([])
  const [activeSid, setActiveSid] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [currentRoute, setCurrentRoute] = useState<string>('')
  const [lastResponseTime, setLastResponseTime] = useState<number | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [streamText, setStreamText] = useState('')
  const [streamSources, setStreamSources] = useState<Source[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)
  const [agentMode, setAgentMode] = useState(false)
  const [toolStatus, setToolStatus] = useState('')

  const msgEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const streamSourcesRef = useRef<Source[]>([])

  const loadSessions = useCallback(async () => {
    try { const r = await listSessions(); setSessions(r.sessions) } catch { /* */ }
  }, [])

  useEffect(() => { loadSessions() }, [loadSessions])
  useEffect(() => { msgEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, streamText])

  const selectSession = async (id: string) => {
    setActiveSid(id)
    try {
      const r = await getSession(id)
      setMessages(r.messages.map((m: MessageInfo) => ({
        role: m.role as 'user' | 'assistant', content: m.content, sources: m.sources,
      })))
    } catch { /* */ }
  }

  const newChat = () => {
    setActiveSid(null); setMessages([]); setStreamText(''); setCurrentRoute(''); setLastResponseTime(null)
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  const delSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('删除该对话？')) return
    try {
      await deleteSession(id)
      if (activeSid === id) newChat()
      await loadSessions()
    } catch { /* */ }
  }

  const send = async () => {
    if (agentMode) {
      return sendAgent()
    }
    if (!input.trim() || streaming) return
    const q = input.trim()
    setInput('')
    setMessages(p => [...p, { role: 'user', content: q }])
    setStreamText(''); setStreamSources([]); streamSourcesRef.current = []; setStreaming(true)
    const t0 = Date.now()

    await askQuestionStream(
      q, activeSid, 'auto', 5,
      (t) => setStreamText(p => p + t),
      (srcs, route) => { setStreamSources(srcs); streamSourcesRef.current = srcs; if (route) setCurrentRoute(route) },
      (newId) => {
        const elapsed = Date.now() - t0
        const finalSources = streamSourcesRef.current
        setStreamText(prev => {
          setMessages(msgs => [...msgs, {
            role: 'assistant', content: prev,
            sources: finalSources.length > 0 ? finalSources : null,
          }])
          return ''
        })
        if (!activeSid) setActiveSid(newId)
        setLastResponseTime(elapsed)
        setStreaming(false)
        loadSessions()
      },
      (err) => { alert('请求失败: ' + err.message); setStreaming(false) },
    )
  }

  const sendAgent = async () => {
    if (!input.trim() || streaming) return
    const q = input.trim()
    setInput('')
    setMessages(p => [...p, { role: 'user', content: q }])
    setStreamText(''); setStreamSources([]); streamSourcesRef.current = []; setToolStatus(''); setStreaming(true)
    const t0 = Date.now()

    await askAgentStream(
      q, activeSid,
      (toolName) => setToolStatus(toolName),
      (t) => setStreamText(p => p + t),
      (srcs) => { setStreamSources(srcs); streamSourcesRef.current = srcs },
      (newId) => {
        const elapsed = Date.now() - t0
        const finalSources = streamSourcesRef.current
        setStreamText(prev => {
          setMessages(msgs => [...msgs, {
            role: 'assistant', content: prev,
            sources: finalSources.length > 0 ? finalSources : null,
          }])
          return ''
        })
        if (!activeSid) setActiveSid(newId)
        setLastResponseTime(elapsed)
        setStreaming(false)
        setToolStatus('')
        loadSessions()
      },
      (err) => { alert('Agent error: ' + err.message); setStreaming(false); setToolStatus('') },
    )
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
  }

  const hasMessages = messages.length > 0 || streaming

  return (
    <div style={page}>
      <h1 style={header}>智能问答</h1>

      <div style={shell}>
        {/* ── Sidebar ── */}
        <div style={sidebar}>
          <div style={sidebarHead}>
            <button style={newBtn} onClick={newChat}>+ 新对话</button>
          </div>
          <div style={sessionList}>
            {sessions.map(s => (
              <div key={s.id} style={sessionRow(s.id === activeSid)} onClick={() => selectSession(s.id)}>
                <span style={sessionTitle}>{s.title}</span>
                <button style={delBtn} onClick={(e) => delSession(s.id, e)} title="删除">×</button>
              </div>
            ))}
          </div>
        </div>

        {/* ── Chat ── */}
        <div style={chat}>
          <div style={msgList}>
            {!hasMessages && (
              <div style={emptyState}>
                <div style={{ fontSize: 40, marginBottom: 12 }}>💬</div>
                <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 4 }}>开始新的对话</div>
                <div style={{ fontSize: 13 }}>输入问题，AI 将基于知识库为你解答</div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} style={msgWrapper}>
                <div style={{ display: 'flex', gap: 10, justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start' }}>
                  {msg.role === 'assistant' && <div style={avatar(false)}>AI</div>}
                  <div style={bubble(msg.role === 'user')}>
                    {msg.role === 'assistant' ? (
                      <ReactMarkdown components={{
                        p: ({ children }) => <p style={markdownStyles.p}>{children}</p>,
                        ul: ({ children }) => <ul style={markdownStyles.ul}>{children}</ul>,
                        ol: ({ children }) => <ol style={markdownStyles.ol}>{children}</ol>,
                        li: ({ children }) => <li style={markdownStyles.li}>{children}</li>,
                        strong: ({ children }) => <strong style={markdownStyles.strong}>{children}</strong>,
                        code: ({ children }) => <code style={markdownStyles.code}>{children}</code>,
                        pre: ({ children }) => <pre style={markdownStyles.pre}>{children}</pre>,
                        blockquote: ({ children }) => <blockquote style={markdownStyles.blockquote}>{children}</blockquote>,
                        h3: ({ children }) => <h3 style={markdownStyles.h3}>{children}</h3>,
                        h4: ({ children }) => <h4 style={markdownStyles.h4}>{children}</h4>,
                      }}>
                        {msg.content}
                      </ReactMarkdown>
                    ) : (
                      msg.content
                    )}
                    {msg.sources && msg.sources.length > 0 && (
                      <SourcesWidget msgIdx={String(i)} sources={msg.sources} expandedKey={expanded} onToggle={setExpanded} />
                    )}
                  </div>
                  {msg.role === 'user' && <div style={avatar(true)}>U</div>}
                </div>
              </div>
            ))}

            {/* Streaming bubble */}
            {streaming && (
              <div style={msgWrapper}>
                <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-start' }}>
                  <div style={avatar(false)}>AI</div>
                  <div style={bubble(false)}>
                    {streamText ? (
                      <ReactMarkdown components={{
                        p: ({ children }) => <p style={markdownStyles.p}>{children}</p>,
                        ul: ({ children }) => <ul style={markdownStyles.ul}>{children}</ul>,
                        ol: ({ children }) => <ol style={markdownStyles.ol}>{children}</ol>,
                        li: ({ children }) => <li style={markdownStyles.li}>{children}</li>,
                        strong: ({ children }) => <strong style={markdownStyles.strong}>{children}</strong>,
                        code: ({ children }) => <code style={markdownStyles.code}>{children}</code>,
                        blockquote: ({ children }) => <blockquote style={markdownStyles.blockquote}>{children}</blockquote>,
                      }}>
                        {streamText}
                      </ReactMarkdown>
                    ) : (
                      <span style={{ color: '#94a3b8', fontStyle: 'italic' }}>思考中...</span>
                    )}
                    {toolStatus && (
                      <div style={{
                        marginTop: 8, padding: '6px 10px', borderRadius: 6,
                        background: '#fef3c7', color: '#92400e',
                        fontSize: 12, fontWeight: 500,
                      }}>
                        {toolStatus}
                      </div>
                    )}
                    {streamSources.length > 0 && streamText.length > 0 && (
                      <SourcesWidget msgIdx="stream" sources={streamSources} expandedKey={expanded} onToggle={setExpanded} />
                    )}
                  </div>
                </div>
              </div>
            )}

            <div ref={msgEndRef} />
          </div>

          {/* Input */}
          <div style={inputRow}>
            <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexShrink: 0 }}>
              {currentRoute && (
                <span style={{
                  padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                  background: currentRoute === 'fast' ? '#dcfce7' :
                              currentRoute === 'precise' ? '#fef9c3' :
                              currentRoute === 'deep' ? '#fee2e2' :
                              '#f1f5f9',
                  color: currentRoute === 'fast' ? '#166534' :
                         currentRoute === 'precise' ? '#854d0e' :
                         currentRoute === 'deep' ? '#991b1b' :
                         '#64748b',
                }}>
                  {currentRoute === 'fast' ? '快速' :
                   currentRoute === 'precise' ? '精准' :
                   currentRoute === 'deep' ? '深度' :
                   currentRoute}
                </span>
              )}
              {lastResponseTime != null && (
                <span style={{
                  padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 500,
                  background: '#f1f5f9', color: '#64748b',
                }}>
                  {lastResponseTime < 1000
                    ? `${lastResponseTime}ms`
                    : `${(lastResponseTime / 1000).toFixed(1)}s`}
                </span>
              )}
              <button
                onClick={() => setAgentMode(m => !m)}
                title={agentMode ? 'Switch to normal' : 'Switch to Agent'}
                style={{
                  padding: '4px 10px', borderRadius: 6, fontSize: 11, fontWeight: 600,
                  cursor: 'pointer', border: 'none',
                  background: agentMode ? '#ede9fe' : '#f1f5f9',
                  color: agentMode ? '#6d28d9' : '#94a3b8',
                  flexShrink: 0,
                }}
              >
                {agentMode ? 'Agent' : 'Normal'}
              </button>
            </div>
            <input
              ref={inputRef}
              style={inputField}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              placeholder="输入问题，Enter 发送..."
              disabled={streaming}
              onFocus={(e) => { e.target.style.borderColor = 'var(--primary)'; e.target.style.background = '#fff' }}
              onBlur={(e) => { e.target.style.borderColor = '#e2e8f0'; e.target.style.background = '#f8fafc' }}
            />
            <button style={sendBtn(!streaming && input.trim().length > 0)} onClick={send} disabled={streaming || !input.trim()}>
              {streaming ? '···' : '发送'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
