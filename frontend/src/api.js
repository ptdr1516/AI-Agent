const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export function getAuthHeaders(extra = {}) {
  const headers = { ...extra };
  const envToken = import.meta.env.VITE_API_TOKEN;
  // Use the unique browser session ID as the tenant token if no static token is enforced
  const localSessionId = localStorage.getItem('nova_user_session_id') || 'anonymous_user';
  const token = envToken || localSessionId;
  
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

async function parseErrorResponse(response) {
  try {
    const err = await response.json();
    const d = err.detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d))
      return d.map((e) => e.msg || JSON.stringify(e)).join("; ");
  } catch (_) {
    /* ignore */
  }
  return response.statusText || "Request failed";
}

export const sendChatMessage = async (sessionId, message) => {
  const response = await fetch(`${API_URL}/chat`, {
    method: "POST",
    headers: getAuthHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ session_id: sessionId, message: message }),
  });
  if (!response.ok) throw new Error(await parseErrorResponse(response));
  return response.json();
};

/**
 * Upload a document for RAG indexing (POST /api/upload).
 * Allowed: .pdf, .txt, .md (see backend).
 */
export const fetchIndexedDocuments = async () => {
  const response = await fetch(`${API_URL}/documents`, {
    headers: getAuthHeaders(),
  });
  if (!response.ok) throw new Error(await parseErrorResponse(response));
  return response.json();
};

export const removeIndexedDocument = async (storedFilename) => {
  const response = await fetch(`${API_URL}/documents/remove`, {
    method: "POST",
    headers: getAuthHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ stored_filename: storedFilename }),
  });
  if (!response.ok) throw new Error(await parseErrorResponse(response));
  return response.json();
};

export const uploadDocument = async (file) => {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${API_URL}/upload`, {
    method: "POST",
    headers: getAuthHeaders(),
    body: form,
  });
  if (!response.ok) throw new Error(await parseErrorResponse(response));
  return response.json();
};

export const streamChatMessage = async (sessionId, message, onToken, onTool) => {
  const response = await fetch(`${API_URL}/chat/stream`, {
    method: "POST",
    headers: getAuthHeaders({ "Content-Type": "application/json" }),
    body: JSON.stringify({ session_id: sessionId, message }),
  });

  if (!response.ok) throw new Error(await parseErrorResponse(response));

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        const dataStr = line.slice(6).trim();
        if (dataStr === "[DONE]") return;
        if (!dataStr) continue;

        try {
          const data = JSON.parse(dataStr);
          if (data.content !== undefined) onToken(data.content);
          if (data.tool && data.status) {
            onTool(data.tool, data.status, data.input, data.output, data.rag_detail);
          }
          if (data.error) throw new Error(data.error);
        } catch (e) {
          console.error("Failed to parse SSE JSON", e, dataStr);
        }
      }
    }
  }
};
