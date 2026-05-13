import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
})

export interface Source {
  content: string
  filename: string
  score: number
}

export interface QuestionResponse {
  answer: string
  sources: Source[]
}

export interface SearchResult {
  content: string
  filename: string
  score: number
}

export interface SearchResponse {
  results: SearchResult[]
}

export interface UploadResponse {
  doc_id: string
  filename: string
  chunks_count: number
}

export interface BatchUploadResult {
  doc_id?: string
  filename: string
  chunks_count?: number
  status: 'ok' | 'error'
  error?: string
}

export interface BatchUploadResponse {
  results: BatchUploadResult[]
  total: number
}

export interface DocumentInfo {
  doc_id: string
  filename: string
  file_size: number
  chunks_count: number
  uploaded_at: string
}

export interface DocumentListResponse {
  documents: DocumentInfo[]
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post<UploadResponse>('/documents/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function uploadDocuments(files: File[]): Promise<BatchUploadResponse> {
  const formData = new FormData()
  files.forEach((file) => formData.append('files', file))
  const { data } = await api.post<BatchUploadResponse>('/documents/upload/batch', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000, // 5 minutes for batch
  })
  return data
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const { data } = await api.get<DocumentListResponse>('/documents')
  return data
}

export async function deleteDocument(docId: string): Promise<void> {
  await api.delete(`/documents/${encodeURIComponent(docId)}`)
}

export async function deleteAllDocuments(): Promise<{ detail: string }> {
  const { data } = await api.delete<{ detail: string }>('/documents')
  return data
}

// === Chunk Preview ===

export interface LeafChunkPreview {
  chunk_index: number
  char_count: number
  preserve: boolean
  undersized: boolean
  content_preview: string
}

export interface ParentChunkPreview {
  id: string
  heading_path: string[]
  char_count: number
  page_start: number | null
  page_end: number | null
  content_preview: string
  leaves: LeafChunkPreview[]
}

export interface ChunkPreviewResponse {
  filename: string
  parents: ParentChunkPreview[]
}

export async function getDocumentChunks(docId: string): Promise<ChunkPreviewResponse> {
  const { data } = await api.get<ChunkPreviewResponse>(`/documents/${encodeURIComponent(docId)}/chunks`)
  return data
}

export async function askQuestion(
  question: string,
  strategy: string = 'hybrid_rerank',
  topK: number = 5,
): Promise<QuestionResponse> {
  const { data } = await api.post<QuestionResponse>('/qa/ask', {
    question,
    strategy,
    top_k: topK,
  })
  return data
}

export async function searchDocuments(
  query: string,
  strategy: string = 'hybrid_rerank',
  topK: number = 5,
): Promise<SearchResponse> {
  const { data } = await api.post<SearchResponse>('/qa/search', {
    query,
    strategy,
    top_k: topK,
  })
  return data
}

// === V2: Session types ===

export interface SessionInfo {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface SessionListResponse {
  sessions: SessionInfo[]
}

export interface MessageInfo {
  role: string
  content: string
  sources: Source[] | null
  created_at: string
}

export interface SessionDetailResponse {
  id: string
  title: string
  messages: MessageInfo[]
}

// === V2: SSE streaming ===

export interface SSEEvent {
  type: 'sources' | 'token' | 'done'
  data: string | Source[]
}

export async function askQuestionStream(
  question: string,
  sessionId: string | null,
  strategy: string,
  topK: number,
  onToken: (token: string) => void,
  onSources: (sources: Source[]) => void,
  onDone: (newSessionId: string) => void,
  onError: (error: Error) => void,
): Promise<void> {
  try {
    const response = await fetch('/api/qa/ask/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        strategy,
        top_k: topK,
        session_id: sessionId,
      }),
    })

    const newSessionId = response.headers.get('X-Session-Id')
    const reader = response.body?.getReader()
    if (!reader) throw new Error('No response body')

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const event: SSEEvent = JSON.parse(line.slice(6))
            if (event.type === 'token') {
              onToken(event.data as string)
            } else if (event.type === 'sources') {
              onSources(event.data as Source[])
            } else if (event.type === 'done') {
              if (newSessionId) onDone(newSessionId)
            }
          } catch {
            // skip parse errors for partial chunks
          }
        }
      }
    }
  } catch (err) {
    onError(err as Error)
  }
}

// === V2: Session API ===

export async function listSessions(): Promise<SessionListResponse> {
  const { data } = await api.get<SessionListResponse>('/qa/sessions')
  return data
}

export async function getSession(sessionId: string): Promise<SessionDetailResponse> {
  const { data } = await api.get<SessionDetailResponse>(`/qa/sessions/${sessionId}`)
  return data
}

export async function deleteSession(sessionId: string): Promise<void> {
  await api.delete(`/qa/sessions/${sessionId}`)
}

// === V3: Evaluation types ===

export interface EvalRunInfo {
  id: string
  strategy: string
  dataset_name: string
  question_count: number
  started_at: string
  completed_at: string | null
  avg_faithfulness: number | null
  avg_context_recall: number | null
  avg_context_precision: number | null
  avg_answer_correctness: number | null
  avg_answer_accuracy: number | null
}

export interface EvalResultItem {
  question: string
  ground_truth: string
  answer: string
  contexts: string[]
  faithfulness: number | null
  context_recall: number | null
  context_precision: number | null
  answer_correctness: number | null
  answer_accuracy: number | null
}

export interface EvalRunDetail extends EvalRunInfo {
  results: EvalResultItem[]
}

export interface EvalListResponse {
  runs: EvalRunInfo[]
}

// === V3: Evaluation API ===

export async function listEvalRuns(): Promise<EvalListResponse> {
  const { data } = await api.get<EvalListResponse>('/eval/results')
  return data
}

export async function getEvalRun(runId: string): Promise<EvalRunDetail> {
  const { data } = await api.get<EvalRunDetail>(`/eval/results/${runId}`)
  return data
}

export async function deleteEvalRun(runId: string): Promise<void> {
  await api.delete(`/eval/results/${encodeURIComponent(runId)}`)
}

export async function triggerEval(strategy: string = 'all'): Promise<{ run_id: string }> {
  const { data } = await api.post<{ run_id: string }>('/eval/run', { strategy })
  return data
}
