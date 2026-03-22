import React, { useState, useCallback, useRef, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import MessageInput from './components/MessageInput';
import ToolStatus from './components/ToolStatus';
import { streamChatMessage, uploadDocument } from './api';
import IndexedDocumentsPanel from './components/IndexedDocumentsPanel';

const generateId = () =>
  (typeof crypto !== 'undefined' && crypto.randomUUID)
    ? crypto.randomUUID()
    : Math.random().toString(36).substring(2, 15);

const STORAGE_KEY = 'nova_conversations';

// A single stable session ID shared across ALL conversations.
// This gives the backend one continuous memory thread for this browser.
// Stored in localStorage so it persists across page reloads and server restarts.
const USER_SESSION_KEY = 'nova_user_session_id';
const getUserSessionId = () => {
  let id = localStorage.getItem(USER_SESSION_KEY);
  if (!id) {
    id = generateId();
    localStorage.setItem(USER_SESSION_KEY, id);
  }
  return id;
};
const USER_SESSION_ID = getUserSessionId();

function mergeRagContext(prev, next) {
  if (!next?.chunks?.length) return prev ?? null;
  if (!prev) {
    return { sources: [...next.sources], chunks: [...next.chunks] };
  }
  const sources = [...new Set([...(prev.sources || []), ...(next.sources || [])])];
  const chunks = [...(prev.chunks || []), ...next.chunks];
  return { sources, chunks };
}

const WELCOME_MSG = {
  role: 'agent',
  content: "Hello! I'm **Nova**, your AI assistant.\n\nI can help you with:\n- 📄 **Your documents** — use the **paperclip** to upload `.pdf`, `.txt`, or `.md`, then ask questions about them\n- 🧮 **Math** — *\"What is 1234 × 56?\"*\n- 🌐 **Web Search** — *\"Tell me about quantum computing\"*\n- 🗄️ **SQL Database** — *\"Who has the highest salary?\"*\n- 🔌 **API Lookups** — *\"Fetch details for user 123\"*\n\nWhat would you like to know today?",
};

const createConversation = () => ({
  id: generateId(),
  // sessionId is stable UUIDv4 — persisted in localStorage so the backend
  // finds the same SQLite history even after a page refresh or server restart
  sessionId: generateId(),
  title: 'New Chat',
  messages: [WELCOME_MSG],
});

const loadConversations = () => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (_) {}
  return null;
};

export default function App() {
  const [conversations, setConversations] = useState(() => {
    // Load persisted conversations so session IDs survive page refresh / server restart
    const saved = loadConversations();
    return saved ?? [createConversation()];
  });
  const [activeId, setActiveId] = useState(() => {
    const saved = loadConversations();
    return saved ? saved[0].id : null;
  });
  const [isTyping, setIsTyping] = useState(false);
  const [activeTool, setActiveTool] = useState(null);
  const activeToolTimeout = useRef(null);
  const [uploadingDoc, setUploadingDoc] = useState(false);
  const [uploadBanner, setUploadBanner] = useState(null);
  const [indexedDocsRevision, setIndexedDocsRevision] = useState(0);

  // Persist every change to localStorage so session IDs are stable across reloads
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (_) {}
  }, [conversations]);

  // If activeId wasn't restored from storage, set it to the first conversation
  useEffect(() => {
    if (!activeId && conversations.length > 0) {
      setActiveId(conversations[0].id);
    }
  }, [activeId, conversations]);

  const activeConv = conversations.find((c) => c.id === activeId);

  const updateConv = useCallback((id, updater) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  }, []);

  const handleNewChat = () => {
    const conv = createConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
  };

  const handleDeleteChat = (idToDelete) => {
    setConversations((prev) => {
      let filtered = prev.filter(c => c.id !== idToDelete);
      
      if (filtered.length === 0) {
        // If we deleted the very last chat, create a brand new empty one
        const newC = createConversation();
        filtered = [newC];
        setActiveId(newC.id);
      } else if (activeId === idToDelete) {
        // If we deleted the active chat, switch to the first available one
        setActiveId(filtered[0].id);
      }
      
      return filtered;
    });
  };

  const streamAgentReply = useCallback(
    async (workingId, userText) => {
      setIsTyping(true);
      setActiveTool(null);

      const onToken = (token) => {
        updateConv(workingId, (c) => {
          const newMessages = [...c.messages];
          const lastMsg = newMessages[newMessages.length - 1];

          if (lastMsg.role === 'user') {
            newMessages.push({ role: 'agent', content: token, reasoningSteps: [] });
          } else {
            const updatedLastMsg = { ...lastMsg, content: lastMsg.content + token };
            newMessages[newMessages.length - 1] = updatedLastMsg;
          }
          return { ...c, messages: newMessages };
        });
      };

      const onTool = (toolName, status, toolInput, toolOutput, ragDetail) => {
        let frontendToolId = null;
        if (toolName.includes('calculator')) frontendToolId = 'calculator';
        else if (toolName.includes('web_search')) frontendToolId = 'search';
        else if (toolName.includes('sql_db')) frontendToolId = 'sql';
        else if (toolName.includes('custom_api')) frontendToolId = 'api';
        else if (toolName.includes('document_search')) frontendToolId = 'rag';

        if (status === 'start') {
          if (activeToolTimeout.current) clearTimeout(activeToolTimeout.current);
          setActiveTool(frontendToolId);
        } else if (status === 'end') {
          activeToolTimeout.current = setTimeout(() => {
            setActiveTool(null);
          }, 800);
        }

        updateConv(workingId, (c) => {
          const newMessages = [...c.messages];
          const lastMsg = newMessages[newMessages.length - 1];

          if (lastMsg.role === 'user') {
            newMessages.push({ role: 'agent', content: '', reasoningSteps: [] });
          }

          const updatedLastMsg = { ...newMessages[newMessages.length - 1] };
          updatedLastMsg.reasoningSteps = [...(updatedLastMsg.reasoningSteps || [])];

          if (status === 'start') {
            updatedLastMsg.reasoningSteps.push({
              tool: toolName,
              input: toolInput,
              output: null,
            });
          } else if (status === 'end') {
            for (let i = updatedLastMsg.reasoningSteps.length - 1; i >= 0; i--) {
              if (updatedLastMsg.reasoningSteps[i].tool === toolName) {
                updatedLastMsg.reasoningSteps[i] = {
                  ...updatedLastMsg.reasoningSteps[i],
                  output: toolOutput,
                };
                break;
              }
            }
            const merged = mergeRagContext(updatedLastMsg.ragContext, ragDetail);
            if (merged) updatedLastMsg.ragContext = merged;
          }

          newMessages[newMessages.length - 1] = updatedLastMsg;
          return { ...c, messages: newMessages };
        });
      };

      try {
        await streamChatMessage(USER_SESSION_ID, userText, onToken, onTool);
      } catch {
        updateConv(workingId, (c) => {
          const newMessages = [...c.messages];
          const last = newMessages[newMessages.length - 1];
          if (last.role === 'agent') {
            last.content = '⚠️ **Connection Error.** Please verify the backend is running.';
          }
          return { ...c, messages: newMessages };
        });
      } finally {
        setIsTyping(false);
        setActiveTool(null);
      }
    },
    [updateConv]
  );

  const handleSend = async (text) => {
    if (!activeConv || isTyping || uploadingDoc) return;
    const workingId = activeId;

    updateConv(workingId, (c) => ({
      ...c,
      title:
        c.messages.filter((m) => m.role === 'user').length === 0
          ? text.length > 36
            ? text.slice(0, 36) + '…'
            : text
          : c.title,
      messages: [...c.messages, { role: 'user', content: text }],
    }));

    await streamAgentReply(workingId, text);
  };

  const handleUpload = async (file) => {
    if (!activeId || isTyping) return;
    setUploadBanner(null);
    setUploadingDoc(true);
    try {
      const data = await uploadDocument(file);
      setUploadBanner({
        ok: true,
        text: `Indexed “${data.filename}” (${data.chunks_indexed} chunk(s)). Syncing to chat…`,
      });

      const workingId = activeId;
      const note = [
        `[Document indexed: "${data.filename}" — ${data.chunks_indexed} chunk(s).`,
        'It is now in the knowledge base. For any question about this file, call document_search first, then answer from the retrieved passages.]',
      ].join(' ');

      updateConv(workingId, (c) => ({
        ...c,
        messages: [...c.messages, { role: 'user', content: note }],
      }));

      await streamAgentReply(workingId, note);
      setUploadBanner({
        ok: true,
        text: `Indexed “${data.filename}” (${data.chunks_indexed} chunk(s)). You can ask about it now.`,
      });
      setIndexedDocsRevision((n) => n + 1);
    } catch (e) {
      setUploadBanner({
        ok: false,
        text: e?.message || 'Upload failed. Check file type (.pdf, .txt, .md) and that the API is running.',
      });
    } finally {
      setUploadingDoc(false);
    }
  };

  return (
    <div className="h-screen w-screen flex relative overflow-hidden bg-bgMain text-textMain antialiased">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={handleNewChat}
        onDelete={handleDeleteChat}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative">
        <Header title={activeConv?.title || 'New Chat'} isTyping={isTyping} />
        
        {/* Messages container */}
        <div className="flex-1 overflow-y-auto relative z-10">
          <ChatWindow messages={activeConv?.messages || []} isTyping={isTyping} />
        </div>
        
        {/* Input box pinned to bottom */}
        <div className="flex-shrink-0 z-20 pb-6 pt-2 bg-gradient-to-t from-bgMain via-bgMain to-transparent">
          {uploadBanner && (
            <div className="max-w-3xl mx-auto px-4 mb-2">
              <div
                className={`text-xs px-3 py-2 rounded-lg border ${
                  uploadBanner.ok
                    ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-200/95'
                    : 'bg-red-950/40 border-red-800/50 text-red-200/95'
                }`}
              >
                {uploadBanner.ok ? uploadBanner.text : `⚠ ${uploadBanner.text}`}
              </div>
            </div>
          )}
          <IndexedDocumentsPanel
            refreshTrigger={indexedDocsRevision}
            disabled={isTyping || uploadingDoc}
          />
          <MessageInput
            onSend={handleSend}
            disabled={isTyping || uploadingDoc}
            onUpload={handleUpload}
            uploading={uploadingDoc}
          />
        </div>
      </div>

      {/* Right Tool Panel - hide on smaller screens */}
      <div className="hidden lg:flex flex-col w-[280px] bg-bgPanel border-l border-bgBorder flex-shrink-0 h-full">
        <div className="p-4 border-b border-bgBorder mt-12">
          <h3 className="text-xs font-semibold text-textMuted uppercase tracking-wider mb-1">Active Tools</h3>
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          <ToolStatus activeTool={activeTool} />
        </div>
        <div className="p-4 border-t border-bgBorder bg-black/20">
           <div className="flex justify-between items-center text-xs text-textFaint">
             <span>Session ID</span>
             <span className="font-mono text-textMuted bg-white/5 px-2 py-0.5 rounded">{activeConv?.sessionId?.slice(-6) || 'none'}</span>
           </div>
        </div>
      </div>
    </div>
  );
}
