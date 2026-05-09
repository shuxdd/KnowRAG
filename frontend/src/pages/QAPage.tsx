import { useState } from 'react'
import { askQuestion, QuestionResponse } from '../api/client'

const pageTitle: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 600,
  marginBottom: 24,
}

const inputArea: React.CSSProperties = {
  display: 'flex',
  gap: 12,
  marginBottom: 24,
  alignItems: 'center',
}

const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: '12px 16px',
  borderRadius: 8,
  border: '1px solid var(--border)',
  fontSize: 15,
  outline: 'none',
}

const selectStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderRadius: 8,
  border: '1px solid var(--border)',
  fontSize: 14,
  background: 'var(--card-bg)',
  cursor: 'pointer',
}

const btnStyle: React.CSSProperties = {
  padding: '12px 28px',
  background: 'var(--primary)',
  color: '#fff',
  border: 'none',
  borderRadius: 8,
  cursor: 'pointer',
  fontSize: 15,
  fontWeight: 500,
}

const answerCard: React.CSSProperties = {
  background: 'var(--card-bg)',
  borderRadius: 12,
  padding: '24px',
  marginBottom: 16,
  lineHeight: 1.8,
  fontSize: 15,
}

const sourceCard: React.CSSProperties = {
  background: 'var(--card-bg)',
  borderRadius: 12,
  padding: '20px 24px',
  marginBottom: 8,
  borderLeft: '3px solid var(--primary)',
}

const strategyLabels: Record<string, string> = {
  vector: '向量检索',
  hybrid: '混合检索',
  hybrid_rerank: '混合检索+Rerank',
}

export default function QAPage() {
  const [question, setQuestion] = useState('')
  const [strategy, setStrategy] = useState('hybrid_rerank')
  const [result, setResult] = useState<QuestionResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const handleAsk = async () => {
    if (!question.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const res = await askQuestion(question, strategy, 5)
      setResult(res)
    } catch (err) {
      console.error('QA failed:', err)
      alert('问答请求失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') handleAsk()
  }

  return (
    <div>
      <h1 style={pageTitle}>智能问答</h1>
      <div style={inputArea}>
        <input
          style={inputStyle}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="输入你的问题..."
        />
        <select
          style={selectStyle}
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
        >
          <option value="vector">向量检索</option>
          <option value="hybrid">混合检索</option>
          <option value="hybrid_rerank">混合检索+Rerank</option>
        </select>
        <button style={btnStyle} onClick={handleAsk} disabled={loading}>
          {loading ? '思考中...' : '提问'}
        </button>
      </div>

      {result && (
        <>
          <div style={answerCard}>
            <div style={{ fontWeight: 600, marginBottom: 8, color: 'var(--text-secondary)' }}>
              回答（{strategyLabels[strategy]}）
            </div>
            <div style={{ whiteSpace: 'pre-wrap' }}>{result.answer}</div>
          </div>
          {result.sources.length > 0 && (
            <div>
              <div style={{
                fontWeight: 600,
                marginBottom: 12,
                color: 'var(--text-secondary)',
                fontSize: 14,
              }}>
                引用来源 ({result.sources.length})
              </div>
              {result.sources.map((src, i) => (
                <div key={i} style={sourceCard}>
                  <div style={{
                    fontSize: 12,
                    color: 'var(--text-secondary)',
                    marginBottom: 6,
                  }}>
                    来源: {src.filename} | 相关度: {src.score.toFixed(4)}
                  </div>
                  <div style={{ fontSize: 14 }}>{src.content}...</div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
