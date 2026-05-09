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

export async function listDocuments(): Promise<DocumentListResponse> {
  const { data } = await api.get<DocumentListResponse>('/documents')
  return data
}

export async function deleteDocument(docId: string): Promise<void> {
  await api.delete(`/documents/${encodeURIComponent(docId)}`)
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
