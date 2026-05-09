import { useState, useEffect, useCallback } from 'react'
import {
  listDocuments,
  uploadDocument,
  deleteDocument,
  DocumentInfo,
} from '../api/client'

const pageTitle: React.CSSProperties = {
  fontSize: 24,
  fontWeight: 600,
  marginBottom: 24,
}

const uploadArea: React.CSSProperties = {
  background: 'var(--card-bg)',
  borderRadius: 12,
  padding: '32px',
  textAlign: 'center',
  border: '2px dashed var(--border)',
  marginBottom: 24,
  cursor: 'pointer',
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

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentInfo[]>([])
  const [loading, setLoading] = useState(false)

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
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true)
    try {
      await uploadDocument(file)
      await fetchDocs()
    } catch (err) {
      console.error('Upload failed:', err)
      alert('上传失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

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

  return (
    <div>
      <h1 style={pageTitle}>文档管理</h1>
      <label style={uploadArea}>
        <input
          type="file"
          accept=".pdf,.docx,.txt,.md"
          onChange={handleUpload}
          disabled={loading}
          style={{ display: 'none' }}
        />
        <div style={{ fontSize: 16, color: 'var(--text-secondary)' }}>
          {loading ? '上传中...' : '点击上传文档 (PDF / DOCX / TXT / MD)'}
        </div>
      </label>

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
    </div>
  )
}
