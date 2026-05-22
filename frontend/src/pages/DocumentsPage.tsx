import { useState, useEffect, useCallback } from 'react'
import {
  listDocuments,
  uploadDocuments,
  deleteDocument,
  deleteAllDocuments,
  getDocumentChunks,
  DocumentInfo,
  ChunkPreviewResponse,
  ParentChunkPreview,
} from '../api/client'

const pageTitle: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 600,
  marginBottom: 24,
}

const uploadArea: React.CSSProperties = {
  display: 'block',
  background: 'var(--card-bg)',
  borderRadius: 12,
  padding: '32px',
  textAlign: 'center',
  border: '2px dashed var(--border)',
  marginBottom: 24,
  cursor: 'pointer',
  boxSizing: 'border-box',
}

const tableStyle: React.CSSProperties = {
  width: '100%',
  background: 'var(--card-bg)',
  borderRadius: 12,
  overflow: 'hidden',
  borderCollapse: 'collapse' as const,
}

const thStyle: React.CSSProperties = {
  textAlign: 'left' as const,
  padding: '12px 16px',
  borderBottom: '1px solid var(--border)',
  color: 'var(--text-secondary)',
  fontSize: 13,
  fontWeight: 600,
}

const tdStyle: React.CSSProperties = {
  padding: '12px 16px',
  borderBottom: '1px solid var(--border)',
  fontSize: 14,
}

const deleteBtn: React.CSSProperties = {
  background: '#ef4444',
  color: '#fff',
  border: 'none',
  padding: '6px 14px',
  borderRadius: 6,
  cursor: 'pointer',
  fontSize: 13,
}

// ── Chunk Preview Modal ──

const modalOverlay: React.CSSProperties = {
  position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
  background: 'rgba(0,0,0,0.4)', display: 'flex',
  alignItems: 'center', justifyContent: 'center', zIndex: 10000,
}

const modalBox: React.CSSProperties = {
  background: '#fff', borderRadius: 14, width: '90vw', maxWidth: 1100,
  maxHeight: '85vh', display: 'flex', flexDirection: 'column',
  boxShadow: '0 8px 30px rgba(0,0,0,0.15)',
}

const modalHeader: React.CSSProperties = {
  padding: '16px 24px', borderBottom: '1px solid var(--border)',
  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
  fontWeight: 600, fontSize: 16,
}

const modalBody: React.CSSProperties = {
  display: 'flex', flex: 1, minHeight: 0, overflow: 'hidden',
}

const leftPanel: React.CSSProperties = {
  width: 320, flexShrink: 0, overflowY: 'auto', borderRight: '1px solid var(--border)',
  padding: 12,
}

const rightPanel: React.CSSProperties = {
  flex: 1, overflowY: 'auto', padding: 16,
}

const parentCard = (active: boolean): React.CSSProperties => ({
  padding: '10px 12px', borderRadius: 8, cursor: 'pointer',
  marginBottom: 6, fontSize: 13,
  background: active ? '#eef2ff' : '#f8fafc',
  border: active ? '1px solid #c7d2fe' : '1px solid transparent',
})

const leafBar: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px',
  borderRadius: 6, marginBottom: 6, fontSize: 13,
  background: '#f8fafc', border: '1px solid #f1f5f9',
}

const tagStyle = (color: string): React.CSSProperties => ({
  display: 'inline-block', padding: '1px 8px', borderRadius: 10,
  fontSize: 11, fontWeight: 600, background: color, color: '#fff',
  flexShrink: 0,
})

function ChunkPreviewModal({ data, onClose }: { data: ChunkPreviewResponse; onClose: () => void }) {
  const [selectedParent, setSelectedParent] = useState<ParentChunkPreview | null>(
    data.parents[0] || null
  )

  return (
    <div style={modalOverlay} onClick={onClose}>
      <div style={modalBox} onClick={(e) => e.stopPropagation()}>
        <div style={modalHeader}>
          <span>{data.filename} — {data.parents.length} 个父块，共 {data.parents.reduce((s, p) => s + p.leaves.length, 0)} 个叶子块</span>
          <button onClick={onClose} style={{
            background: 'none', border: 'none', fontSize: 20, cursor: 'pointer',
            color: '#94a3b8', padding: '4px 8px',
          }}>x</button>
        </div>
        <div style={modalBody}>
          {/* Left: parent list */}
          <div style={leftPanel}>
            {data.parents.map((p) => (
              <div
                key={p.id}
                style={parentCard(selectedParent?.id === p.id)}
                onClick={() => setSelectedParent(p)}
              >
                <div style={{ fontWeight: 600, marginBottom: 4 }}>
                  {p.heading_path.length > 0 ? p.heading_path.join(' > ') : '(无标题)'}
                </div>
                <div style={{ color: '#94a3b8', fontSize: 12 }}>
                  {p.char_count} 字 · {p.leaves.length} 个叶子
                  {p.page_start != null && ` · p${p.page_start}${p.page_end != null && p.page_end !== p.page_start ? `-${p.page_end}` : ''}`}
                  {p.created_at && ` · ${new Date(p.created_at).toLocaleDateString('zh-CN')}`}
                </div>
                <div style={{ color: '#64748b', fontSize: 12, marginTop: 4, lineHeight: 1.4 }}>
                  {p.content_preview}...
                </div>
              </div>
            ))}
          </div>

          {/* Right: leaf list */}
          <div style={rightPanel}>
            {selectedParent ? (
              <>
                <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 6 }}>
                  {selectedParent.heading_path.length > 0
                    ? selectedParent.heading_path.join(' > ')
                    : '(无标题)'}
                </div>
                <div style={{ color: '#94a3b8', fontSize: 12, marginBottom: 16 }}>
                  {selectedParent.char_count} 字 · 页 {selectedParent.page_start ?? '?'}-{selectedParent.page_end ?? '?'}
                  {selectedParent.created_at && ` · ${new Date(selectedParent.created_at).toLocaleDateString('zh-CN')}`}
                </div>
                {selectedParent.leaves.map((leaf) => (
                  <div
                    key={leaf.chunk_index}
                    style={{
                      ...leafBar,
                      borderColor: leaf.undersized ? '#fecaca' : undefined,
                      background: leaf.undersized ? '#fef2f2' : leaf.preserve ? '#fefce8' : undefined,
                    }}
                  >
                    <span style={{ fontWeight: 600, minWidth: 24, color: '#64748b' }}>
                      #{leaf.chunk_index}
                    </span>
                    <span style={tagStyle(leaf.preserve ? '#eab308' : leaf.undersized ? '#ef4444' : '#3b82f6')}>
                      {leaf.preserve ? '保留' : leaf.undersized ? '过小' : `${leaf.char_count}字`}
                    </span>
                    <span style={{ color: '#334155', lineHeight: 1.4, flex: 1 }}>
                      {leaf.content_preview}{leaf.char_count > 150 ? '...' : ''}
                    </span>
                  </div>
                ))}
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: 60, color: '#94a3b8' }}>
                选择一个父块查看叶子详情
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState('')

  const fetchDocs = useCallback(async () => {
    try {
      const res = await listDocuments()
      setDocs(res.documents)
    } catch (err) {
      console.error('Failed to fetch documents:', err)
    }
  }, [])

  useEffect(() => {
    fetchDocs()
  }, [fetchDocs])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (files.length === 0) return
    setLoading(true)
    setUploadMsg(`上传中... (0/${files.length})`)
    try {
      const res = await uploadDocuments(files)
      const ok = res.results.filter((r) => r.status === 'ok').length
      const failed = res.results.filter((r) => r.status === 'error')
      if (failed.length > 0) {
        const msgs = failed.map((f) => `${f.filename}: ${f.error}`).join('\n')
        alert(`部分上传失败:\n${msgs}`)
      }
      setUploadMsg(`完成: ${ok}/${files.length} 个文件上传成功`)
      await fetchDocs()
      // reset file input so same file can be re-uploaded
      e.target.value = ''
    } catch (err) {
      console.error('Upload failed:', err)
      alert('上传失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
      setTimeout(() => setUploadMsg(''), 3000)
    }
  }

  const [chunkPreview, setChunkPreview] = useState<ChunkPreviewResponse | null>(null)
  const [previewLoading, setPreviewLoading] = useState(false)

  const handleDelete = async (docId: string) => {
    if (!confirm('确定删除此文档？')) return
    try {
      await deleteDocument(docId)
      await fetchDocs()
    } catch (err) {
      console.error('Delete failed:', err)
      alert('删除失败')
    }
  }

  const handlePreview = async (docId: string) => {
    setPreviewLoading(true)
    try {
      const res = await getDocumentChunks(docId)
      setChunkPreview(res)
    } catch (err) {
      console.error('Preview failed:', err)
      alert('获取分块预览失败')
    } finally {
      setPreviewLoading(false)
    }
  }

  const handleDeleteAll = async () => {
    if (!confirm('确定要删除知识库中所有文档吗？此操作不可恢复！')) return
    if (!confirm('再次确认：将删除所有文档及其分块数据')) return
    try {
      const res = await deleteAllDocuments()
      alert(res.detail)
      await fetchDocs()
    } catch (err) {
      console.error('Delete all failed:', err)
      alert('清空失败')
    }
  }

  return (
    <div>
      <h1 style={pageTitle}>文档管理</h1>
      <label style={uploadArea}>
        <input
          type="file"
          accept=".pdf,.docx,.txt,.md"
          onChange={handleUpload}
          disabled={loading}
          multiple
          style={{ display: 'none' }}
        />
        <div style={{ fontSize: 16, color: 'var(--text-secondary)' }}>
          {loading ? uploadMsg || '上传中...' : '点击上传文档 (支持多选: PDF / DOCX / TXT / MD)'}
        </div>
      </label>

      {docs.length > 0 && (
        <div style={{ marginBottom: 16, display: 'flex', justifyContent: 'flex-end' }}>
          <button
            onClick={handleDeleteAll}
            style={{
              background: '#fff', color: '#ef4444', border: '1px solid #fecaca',
              padding: '8px 20px', borderRadius: 8, cursor: 'pointer', fontSize: 13,
              fontWeight: 500,
            }}
          >
            一键清空 ({docs.length} 个文档)
          </button>
        </div>
      )}

      {docs.length === 0 ? (
        <div style={{ textAlign: 'center', padding: 48, color: 'var(--text-secondary)' }}>
          暂无文档，请上传
        </div>
      ) : (
        <table style={tableStyle}>
          <thead>
            <tr>
              <th style={thStyle}>文件名</th>
              <th style={thStyle}>大小</th>
              <th style={thStyle}>分块数</th>
              <th style={thStyle}>操作</th>
            </tr>
          </thead>
          <tbody>
            {docs.map((doc) => (
              <tr key={doc.doc_id}>
                <td style={tdStyle}>{doc.filename}</td>
                <td style={tdStyle}>{(doc.file_size / 1024).toFixed(1)} KB</td>
                <td style={tdStyle}>{doc.chunks_count}</td>
                <td style={tdStyle}>
                  <button
                    style={{ ...deleteBtn, background: '#3b82f6', marginRight: 8 }}
                    onClick={() => handlePreview(doc.doc_id)}
                  >
                    预览分块
                  </button>
                  <button
                    style={deleteBtn}
                    onClick={() => handleDelete(doc.doc_id)}
                  >
                    删除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {/* Chunk Preview Modal */}
      {chunkPreview && (
        <ChunkPreviewModal
          data={chunkPreview}
          onClose={() => setChunkPreview(null)}
        />
      )}
      {previewLoading && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
          background: 'rgba(0,0,0,0.3)', display: 'flex',
          alignItems: 'center', justifyContent: 'center', zIndex: 9999,
        }}>
          <div style={{ background: '#fff', padding: '24px 40px', borderRadius: 12, fontSize: 15 }}>
            加载中...
          </div>
        </div>
      )}
    </div>
  )
}
