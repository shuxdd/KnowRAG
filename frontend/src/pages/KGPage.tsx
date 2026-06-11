import { useState, useEffect } from 'react'
import {
  getKGStats, listKGEntities, getKGEntityDetail,
  type KGStats, type KGEntitySummary, type KGEntityDetail,
} from '../api/client'

export default function KGPage() {
  const [stats, setStats] = useState<KGStats | null>(null)
  const [entities, setEntities] = useState<KGEntitySummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detail, setDetail] = useState<KGEntityDetail | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getKGStats().then(setStats).catch(() => {})
  }, [])

  useEffect(() => {
    setLoading(true)
    listKGEntities({ search: search || undefined, page, page_size: 20 })
      .then((res) => { setEntities(res.entities); setTotal(res.total) })
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [search, page])

  useEffect(() => {
    if (selectedId) {
      getKGEntityDetail(selectedId).then(setDetail).catch(() => setDetail(null))
    } else {
      setDetail(null)
    }
  }, [selectedId])

  return (
    <div style={{ display: 'flex', gap: 24, height: 'calc(100vh - 64px)' }}>
      {/* Left: entity list */}
      <div style={{ width: 360, display: 'flex', flexDirection: 'column' }}>
        <h2 style={{ margin: '0 0 16px' }}>知识图谱</h2>
        {stats && (
          <div style={{ display: 'flex', gap: 16, marginBottom: 16, fontSize: 13, color: '#888' }}>
            <span>实体: {stats.entity_count}</span>
            <span>关系: {stats.relation_count}</span>
            <span>类型: {stats.type_count}</span>
          </div>
        )}
        <input
          placeholder="搜索实体..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1) }}
          style={{ padding: '8px 12px', border: '1px solid #ddd', borderRadius: 6, marginBottom: 12 }}
        />
        <div style={{ flex: 1, overflow: 'auto', border: '1px solid #eee', borderRadius: 6 }}>
          {loading ? (
            <div style={{ padding: 16, textAlign: 'center', color: '#999' }}>加载中...</div>
          ) : entities.length === 0 ? (
            <div style={{ padding: 16, textAlign: 'center', color: '#999' }}>暂无实体</div>
          ) : (
            entities.map((e) => (
              <div
                key={e.id || e.name}
                onClick={() => setSelectedId(e.id)}
                style={{
                  padding: '10px 14px', cursor: 'pointer', borderBottom: '1px solid #f0f0f0',
                  background: selectedId === e.id ? '#f0f7ff' : 'transparent',
                }}
              >
                <div style={{ fontWeight: 500 }}>{e.name}</div>
                {e.type && <span style={{ fontSize: 12, color: '#888' }}>{e.type}</span>}
                {e.description && <div style={{ fontSize: 12, color: '#666', marginTop: 2 }}>{e.description}</div>}
              </div>
            ))
          )}
        </div>
        {total > 20 && (
          <div style={{ display: 'flex', justifyContent: 'center', gap: 8, marginTop: 8 }}>
            <button disabled={page <= 1} onClick={() => setPage(page - 1)}>上一页</button>
            <span style={{ lineHeight: '32px', fontSize: 13 }}>{page} / {Math.ceil(total / 20)}</span>
            <button disabled={page * 20 >= total} onClick={() => setPage(page + 1)}>下一页</button>
          </div>
        )}
      </div>

      {/* Right: entity detail */}
      <div style={{ flex: 1, overflow: 'auto' }}>
        {!detail ? (
          <div style={{ padding: 40, textAlign: 'center', color: '#999' }}>
            选择左侧实体查看详情
          </div>
        ) : (
          <div>
            <h3 style={{ marginTop: 0 }}>{detail.entity.name}</h3>
            {detail.entity.type && <div style={{ color: '#888', marginBottom: 4 }}>类型: {detail.entity.type}</div>}
            {detail.entity.description && <div style={{ marginBottom: 16 }}>{detail.entity.description}</div>}

            {detail.relations.length > 0 && (
              <>
                <h4>关系</h4>
                <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 16 }}>
                  <thead>
                    <tr style={{ borderBottom: '2px solid #eee', textAlign: 'left' }}>
                      <th style={{ padding: 8 }}>方向</th>
                      <th style={{ padding: 8 }}>关系</th>
                      <th style={{ padding: 8 }}>关联实体</th>
                      <th style={{ padding: 8 }}>上下文</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.relations.map((r, i) => (
                      <tr key={i} style={{ borderBottom: '1px solid #f0f0f0' }}>
                        <td style={{ padding: 8 }}>{r.direction === 'outgoing' ? '→' : '←'}</td>
                        <td style={{ padding: 8 }}>{r.relation}</td>
                        <td style={{ padding: 8 }}>
                          <button
                            style={{ background: 'none', border: 'none', color: '#1677ff', cursor: 'pointer', padding: 0 }}
                            onClick={() => setSelectedId((r.target || r.source)?.id || null)}
                          >
                            {(r.target || r.source)?.name}
                          </button>
                        </td>
                        <td style={{ padding: 8, fontSize: 12, color: '#666', maxWidth: 300 }}>
                          {r.context || '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}

            {detail.mentioned_in.length > 0 && (
              <>
                <h4>来源文档</h4>
                {detail.mentioned_in.map((c, i) => (
                  <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #f5f5f5', fontSize: 13 }}>
                    <span style={{ fontWeight: 500 }}>{c.filename}</span>
                    {c.heading_path && <span style={{ color: '#888', marginLeft: 8 }}>{c.heading_path}</span>}
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
