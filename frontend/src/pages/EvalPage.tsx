import { useState, useEffect, useCallback } from 'react'
import {
  listEvalRuns, getEvalRun, triggerEval, deleteEvalRun,
  EvalRunInfo, EvalRunDetail,
} from '../api/client'

// ── Styles ──

const page: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', height: 'calc(100vh - 88px)',
}

const header: React.CSSProperties = {
  fontSize: 22, fontWeight: 700, marginBottom: 16, color: '#1e293b',
}

const shell: React.CSSProperties = {
  display: 'flex', gap: 16, flex: 1, minHeight: 0,
}

const panel: React.CSSProperties = {
  width: 280, flexShrink: 0, background: '#fff', borderRadius: 12,
  boxShadow: '0 1px 3px rgba(0,0,0,.06)', overflowY: 'auto', padding: 16,
}

const runBtn: React.CSSProperties = {
  width: '100%', padding: '10px 0', background: 'var(--primary)',
  color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer',
  fontSize: 13, fontWeight: 600, marginBottom: 12,
}

const runItem = (active: boolean): React.CSSProperties => ({
  padding: '10px 12px', borderRadius: 8, cursor: 'pointer',
  fontSize: 13, marginBottom: 4,
  background: active ? '#eef2ff' : 'transparent',
  fontWeight: active ? 600 : 400,
})

const detail: React.CSSProperties = {
  flex: 1, background: '#fff', borderRadius: 12,
  boxShadow: '0 1px 3px rgba(0,0,0,.06)', overflowY: 'auto', padding: 24,
}

const tbl: React.CSSProperties = {
  width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 32,
}

const th: React.CSSProperties = {
  textAlign: 'left', padding: '10px 14px', borderBottom: '2px solid #e2e8f0',
  fontWeight: 600, color: '#475569', fontSize: 12, textTransform: 'uppercase',
  letterSpacing: '.5px',
}

const td = (isBest: boolean): React.CSSProperties => ({
  padding: '10px 14px', borderBottom: '1px solid #f1f5f9',
  fontWeight: isBest ? 700 : 400,
  color: isBest ? '#4338ca' : '#1e293b',
  background: isBest ? '#eef2ff' : 'transparent',
})

const stratLabel: React.CSSProperties = {
  display: 'inline-block', padding: '2px 10px', borderRadius: 12,
  fontSize: 11, fontWeight: 600,
}

const badgeColors: Record<string, React.CSSProperties> = {
  vector: { background: '#fef3c7', color: '#92400e' },
  hybrid: { background: '#dbeafe', color: '#1e40af' },
  hybrid_rerank: { background: '#dcfce7', color: '#166534' },
}

const emptyState: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', alignItems: 'center',
  justifyContent: 'center', height: '100%', color: '#94a3b8', fontSize: 14,
}

// ── Radar Chart (pure SVG) ──

const METRICS = [
  { key: 'avg_faithfulness', label: 'Faithfulness' },
  { key: 'avg_context_recall', label: 'Ctx Recall' },
  { key: 'avg_context_precision', label: 'Ctx Precision' },
  { key: 'avg_answer_relevancy', label: 'Relevancy' },
]
const M_KEYS = METRICS.map(m => m.key)

function RadarChart({ runs }: { runs: EvalRunInfo[] }) {
  const size = 280; const cx = size / 2; const cy = size / 2; const r = 100
  const n = METRICS.length

  const runsWithData = runs.filter(r => M_KEYS.every(k => (r as any)[k] != null))
  if (runsWithData.length === 0) return null

  const colors: Record<string, string> = {
    vector: '#f59e0b', hybrid: '#3b82f6', hybrid_rerank: '#10b981',
  }

  const angle = (i: number) => (Math.PI * 2 * i) / n - Math.PI / 2
  const px = (i: number, v: number) => cx + r * v * Math.cos(angle(i))
  const py = (i: number, v: number) => cy + r * v * Math.sin(angle(i))

  const gridLevels = [0.2, 0.4, 0.6, 0.8, 1.0]

  return (
    <div>
      <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Radar Chart</h3>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Grid polygons */}
        {gridLevels.map(lvl => (
          <polygon key={lvl} points={METRICS.map((_, i) => `${px(i, lvl)},${py(i, lvl)}`).join(' ')} fill="none" stroke="#e2e8f0" strokeWidth={1} />
        ))}
        {/* Axis lines */}
        {METRICS.map((_, i) => (
          <line key={i} x1={cx} y1={cy} x2={px(i, 1)} y2={py(i, 1)} stroke="#e2e8f0" strokeWidth={1} />
        ))}
        {/* Data polygons */}
        {runsWithData.map(run => (
          <polygon key={run.id} points={METRICS.map((m, i) => `${px(i, (run as any)[m.key] || 0)},${py(i, (run as any)[m.key] || 0)}`).join(' ')} fill={colors[run.strategy] || '#888'} fillOpacity={0.12} stroke={colors[run.strategy] || '#888'} strokeWidth={2} />
        ))}
        {/* Labels */}
        {METRICS.map((m, i) => (
          <text key={i} x={px(i, 1.2)} y={py(i, 1.2)} textAnchor="middle" dominantBaseline="middle" fontSize={10} fill="#64748b" fontWeight={500}>
            {m.label}
          </text>
        ))}
      </svg>
      {/* Legend */}
      <div style={{ display: 'flex', gap: 16, marginTop: 12, justifyContent: 'center' }}>
        {runsWithData.map(run => (
          <div key={run.id} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: colors[run.strategy] || '#888' }} />
            {run.strategy}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Page ──

export default function EvalPage() {
  const [runs, setRuns] = useState<EvalRunInfo[]>([])
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [detailData, setDetailData] = useState<EvalRunDetail | null>(null)
  const [running, setRunning] = useState(false)

  const loadRuns = useCallback(async () => {
    try { const r = await listEvalRuns(); setRuns(r.runs) } catch { /* */ }
  }, [])

  useEffect(() => { loadRuns() }, [loadRuns])

  // Auto-refresh while running
  useEffect(() => {
    if (!running) return
    const t = setInterval(loadRuns, 3000)
    return () => clearInterval(t)
  }, [running, loadRuns])

  const handleTrigger = async () => {
    setRunning(true)
    try {
      await triggerEval('all')
      await loadRuns()
    } catch (err) {
      alert('触发评估失败: ' + (err as Error).message)
    }
    setRunning(false)
  }

  const handleSelect = async (runId: string) => {
    setSelectedRun(runId)
    try {
      const d = await getEvalRun(runId)
      setDetailData(d)
    } catch { /* */ }
  }

  const handleDelete = async (runId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (!confirm('确定删除此评估报告？')) return
    try {
      await deleteEvalRun(runId)
      if (selectedRun === runId) {
        setSelectedRun(null)
        setDetailData(null)
      }
      await loadRuns()
    } catch (err) {
      alert('删除失败: ' + (err as Error).message)
    }
  }

  // Group latest runs by strategy for comparison table
  const latestByStrategy: Record<string, EvalRunInfo> = {}
  for (const r of runs) {
    if (!latestByStrategy[r.strategy]) latestByStrategy[r.strategy] = r
  }
  const compareRuns = Object.values(latestByStrategy)

  return (
    <div style={page}>
      <h1 style={header}>评估报告</h1>

      <div style={shell}>
        {/* Left sidebar */}
        <div style={panel}>
          <button style={runBtn} onClick={handleTrigger} disabled={running}>
            {running ? '评估中...' : '▶ 开始评估'}
          </button>
          <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 8 }}>
            全部策略 · {running ? '运行中' : `${runs.length} 次历史`}
          </div>
          {runs.map(r => (
            <div key={r.id} style={runItem(selectedRun === r.id)} onClick={() => handleSelect(r.id)}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ ...badgeColors[r.strategy] || badgeColors.vector, ...stratLabel }}>
                  {r.strategy}
                </span>
                <span style={{ fontSize: 11, color: '#94a3b8' }}>
                  {r.completed_at ? new Date(r.completed_at).toLocaleDateString() : 'running...'}
                </span>
                <button
                  onClick={(e) => handleDelete(r.id, e)}
                  style={{
                    marginLeft: 'auto', background: 'none', border: 'none',
                    color: '#cbd5e1', cursor: 'pointer', fontSize: 16,
                    padding: '0 4px', lineHeight: 1,
                  }}
                  title="删除评估报告"
                  onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = '#cbd5e1')}
                >
                  x
                </button>
              </div>
            </div>
          ))}
        </div>

        {/* Main content */}
        <div style={detail}>
          {detailData ? (
            <>
              {/* Per-run detail */}
              <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12 }}>
                {detailData.strategy} — {detailData.question_count} questions
              </h3>
              <table style={tbl}>
                <thead>
                  <tr>
                    <th style={th}>Metric</th>
                    <th style={th}>Score</th>
                  </tr>
                </thead>
                <tbody>
                  {METRICS.map(m => (
                    <tr key={m.key}>
                      <td style={td(false)}>{m.label}</td>
                      <td style={td(false)}>
                        {(detailData as any)[m.key] != null ? ((detailData as any)[m.key] * 100).toFixed(1) + '%' : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>

              {/* Cross-strategy comparison */}
              {compareRuns.length > 1 && (
                <>
                  <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 12, marginTop: 24 }}>
                    Strategy Comparison
                  </h3>
                  <table style={tbl}>
                    <thead>
                      <tr>
                        <th style={th}>Metric</th>
                        {compareRuns.map(r => (
                          <th key={r.id} style={th}>
                            <span style={{ ...badgeColors[r.strategy] || badgeColors.vector, ...stratLabel }}>
                              {r.strategy}
                            </span>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {METRICS.map(m => {
                        const best = Math.max(...compareRuns.map(r => (r as any)[m.key] || 0))
                        return (
                          <tr key={m.key}>
                            <td style={td(false)}>{m.label}</td>
                            {compareRuns.map(r => {
                              const val = (r as any)[m.key]
                              const isBest = val != null && val === best && best > 0
                              return (
                                <td key={r.id} style={td(isBest)}>
                                  {val != null ? (val * 100).toFixed(1) + '%' : 'N/A'}
                                </td>
                              )
                            })}
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                  <RadarChart runs={compareRuns} />
                </>
              )}
            </>
          ) : (
            <div style={emptyState}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>📊</div>
              <div style={{ fontSize: 15, fontWeight: 500, marginBottom: 4 }}>选择一次评估查看详情</div>
              <div style={{ fontSize: 13 }}>或点击「开始评估」运行新的评估</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
