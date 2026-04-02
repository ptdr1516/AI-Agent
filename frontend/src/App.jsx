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
  messages: [],
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
  const [searchTerm, setSearchTerm] = useState('');
  const [indexedDocsRevision, setIndexedDocsRevision] = useState(0);
  const [isSidebarOpen, setIsSidebarOpen] = useState(() => {
    const saved = localStorage.getItem('nova_sidebar_open');
    if (saved !== null) return saved === 'true';
    return window.innerWidth >= 768; // Default open on desktop
  });
  const [editingMessage, setEditingMessage] = useState(null);
  const [editingConversationId, setEditingConversationId] = useState(null);

  useEffect(() => {
    localStorage.setItem('nova_sidebar_open', isSidebarOpen);
  }, [isSidebarOpen]);

  // Clear editing state when switching conversations to prevent cross-contamination
  useEffect(() => {
    setEditingMessage(null);
    setEditingConversationId(null);
  }, [activeId]);

  const [isToolPanelOpen, setIsToolPanelOpen] = useState(false);

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

  // Close sidebar on mobile when selecting a chat
  const handleSelectChat = (id) => {
    setActiveId(id);
    setIsSidebarOpen(false);
  };

  const activeConv = conversations.find((c) => c.id === activeId);

  const updateConv = useCallback((id, updater) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  }, []);

  const deleteMessage = (convId, msgId) => {
    updateConv(convId, (c) => ({
      ...c,
      messages: c.messages.filter((m) => m.id !== msgId),
    }));
  };

  const renameConversation = (id, newTitle) => {
    updateConv(id, (c) => ({ ...c, title: newTitle }));
  };

  const togglePin = (id) => {
    updateConv(id, (c) => ({ ...c, isPinned: !c.isPinned }));
  };

  const toggleArchive = (id) => {
    updateConv(id, (c) => ({ ...c, isArchived: !c.isArchived }));
  };

  const toggleMessagePin = (convId, msgId) => {
    updateConv(convId, (c) => ({
      ...c,
      messages: c.messages.map((m) =>
        m.id === msgId ? { ...m, isPinned: !m.isPinned } : m
      ),
    }));
  };

  const handleRetry = async (convId) => {
    const conv = conversations.find((c) => c.id === convId);
    if (!conv || isTyping) return;

    // Find the last user message to retry from
    const userMsgs = conv.messages.filter((m) => m.role === 'user');
    if (userMsgs.length === 0) return;
    
    const lastUserText = userMsgs[userMsgs.length - 1].content;

    // Remove any trailing agent error/partial messages after that user message
    const lastUserIdx = conv.messages.findLastIndex((m) => m.role === 'user');
    updateConv(convId, (c) => ({
      ...c,
      messages: c.messages.slice(0, lastUserIdx + 1),
    }));

    await streamAgentReply(convId, lastUserText);
  };

  const handleEditMessage = (message) => {
    setEditingMessage(message);
    setEditingConversationId(activeId);  // Track which conversation we're editing in
  };

  const handleCancelEdit = () => {
    setEditingMessage(null);
    setEditingConversationId(null);  // Clear conversation tracking
  };

  const handleNewChat = () => {
    const conv = createConversation();
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
  };

  const deleteConversation = (idToDelete) => {
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
            newMessages.push({ 
              id: generateId(),
              role: 'agent', 
              content: token, 
              reasoningSteps: [],
              timestamp: Date.now()
            });
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

        // Map to friendly display name for reasoning steps
        const getToolDisplayName = (name) => {
          if (name.includes('calculator')) return 'Calculator';
          if (name.includes('web_search')) return 'Web Search';
          if (name.includes('sql_db')) return 'SQL Database';
          if (name.includes('custom_api')) return 'User Lookups';
          if (name.includes('document_search')) return 'Document Search';
          return name.split('_').pop() || name; // fallback
        };

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
            newMessages.push({ 
              id: generateId(),
              role: 'agent', 
              content: '', 
              reasoningSteps: [],
              timestamp: Date.now()
            });
          }

          const updatedLastMsg = { ...newMessages[newMessages.length - 1] };
          updatedLastMsg.reasoningSteps = [...(updatedLastMsg.reasoningSteps || [])];

          if (status === 'start') {
            updatedLastMsg.reasoningSteps.push({
              tool: getToolDisplayName(toolName),
              input: toolInput,
              output: null,
            });
          } else if (status === 'end') {
            const displayName = getToolDisplayName(toolName);
            for (let i = updatedLastMsg.reasoningSteps.length - 1; i >= 0; i--) {
              if (updatedLastMsg.reasoningSteps[i].tool === displayName) {
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
        const conv = conversations.find(c => c.id === workingId);
        const sid = conv?.sessionId || USER_SESSION_ID; // Fallback to global if somehow missing
        await streamChatMessage(sid, userText, onToken, onTool);
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

    // If we're editing a message, remove all messages after the edited one and update it
    if (editingMessage) {
      updateConv(workingId, (c) => {
        const msgIndex = c.messages.findIndex(m => m.id === editingMessage.id);
        if (msgIndex === -1) return c;
        
        // Keep messages up to and including the edited message, but update its content
        const updatedMessages = c.messages.slice(0, msgIndex);
        updatedMessages.push({
          ...editingMessage,
          content: text,
          timestamp: Date.now()
        });
        
        return {
          ...c,
          messages: updatedMessages
        };
      });
      
      setEditingMessage(null);
      await streamAgentReply(workingId, text);
      return;
    }

    // Normal send flow
    updateConv(workingId, (c) => ({
      ...c,
      title:
        c.messages.filter((m) => m.role === 'user').length === 0
          ? text.length > 36
            ? text.slice(0, 36) + '…'
            : text
          : c.title,
      messages: [...c.messages, { 
        id: generateId(),
        role: 'user', 
        content: text,
        timestamp: Date.now()
      }],
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
        title: c.messages.filter(m => m.role === 'user').length === 0 ? data.filename : c.title,
        messages: [...c.messages, { 
          id: generateId(),
          role: 'user', 
          content: note,
          timestamp: Date.now()
        }],
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

  // Apply Search Filtering
  const filteredMessages = (activeConv?.messages || []).filter(m => 
    !searchTerm || m.content.toLowerCase().includes(searchTerm.toLowerCase())
  );

  // Get the Reasoning Steps from the very last message if it's an agent message
  const activeReasoning = (activeConv?.messages && activeConv.messages.length > 0)
    ? activeConv.messages[activeConv.messages.length - 1].reasoningSteps || []
    : [];

  return (
    <div className="h-screen w-full flex overflow-hidden bg-bgMain text-textMain antialiased transition-all duration-300">
      {/* Mobile Overlay */}
      {(isSidebarOpen || isToolPanelOpen) && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden transition-opacity"
          onClick={() => { setIsSidebarOpen(false); setIsToolPanelOpen(false); }}
        />
      )}

      <Sidebar
      conversations={conversations}
      activeId={activeId}
      onSelect={setActiveId}
      onNew={handleNewChat}
      onDelete={deleteConversation}
      onRename={renameConversation}
      onPin={togglePin}
      onArchive={toggleArchive}
      isOpen={isSidebarOpen}
      onClose={() => setIsSidebarOpen(false)}
    />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0 h-full relative">
        <Header            activeTitle={activeConv?.title || 'New Chat'}
            onMenuClick={() => setIsSidebarOpen(true)}
            onToggleSidebar={() => setIsSidebarOpen(prev => !prev)}
            onToolsClick={() => setIsToolPanelOpen(true)}
            searchTerm={searchTerm}
            onSearchChange={setSearchTerm}
            isOpen={isSidebarOpen}
          />
        
        {/* Messages container - strictly constrained scroll area */}
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden relative z-10 custom-scrollbar">
          <ChatWindow 
            messages={filteredMessages} 
            isTyping={isTyping} 
            onDelete={(mid) => deleteMessage(activeId, mid)}
            onRetry={() => handleRetry(activeId)}
            onPinMessage={(mid) => toggleMessagePin(activeId, mid)}
            searchTerm={searchTerm}
            onEditMessage={handleEditMessage}
          />
        </div>
        
        {/* Input box pinned to bottom */}
        <div className="flex-shrink-0 z-20 pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-2 w-full bg-gradient-to-t from-bgMain via-bgMain to-transparent">
          {uploadBanner && (
            <div className="max-w-3xl mx-auto px-4 mb-2 relative">
              <div
                className={`text-xs px-3 py-2 pr-8 rounded-lg border relative ${
                  uploadBanner.ok
                    ? 'bg-emerald-950/40 border-emerald-800/50 text-emerald-200/95'
                    : 'bg-red-950/40 border-red-800/50 text-red-200/95'
                }`}
              >
                {uploadBanner.ok ? uploadBanner.text : `⚠ ${uploadBanner.text}`}
                <button 
                  onClick={() => setUploadBanner(null)}
                  className="absolute right-2 top-1.5 p-1 hover:bg-white/10 rounded transition-colors"
                  aria-label="Close banner"
                >
                  <svg className="w-3.5 h-3.5 opacity-60" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
          )}
          <IndexedDocumentsPanel
            refreshTrigger={indexedDocsRevision}
            disabled={isTyping || uploadingDoc}
            onRemove={() => setUploadBanner(null)}
          />
          <MessageInput
            onSend={handleSend}
            disabled={isTyping || uploadingDoc}
            onUpload={handleUpload}
            uploading={uploadingDoc}
            editMessage={editingConversationId === activeId ? editingMessage : null}  // Only show edit if in correct conversation
            onCancelEdit={handleCancelEdit}
          />
        </div>
      </div>

      {/* Right Tool Panel */}
      <div className={`
        fixed inset-y-0 right-0 z-50 xl:relative xl:z-auto h-full w-4/5 sm:w-72 xl:w-72 flex-shrink-0 flex flex-col bg-bgPanel border-l border-bgBorder transition-transform duration-300
        ${isToolPanelOpen ? 'translate-x-0 shadow-[-10px_0_20px_rgba(0,0,0,0.5)] xl:shadow-none' : 'translate-x-full xl:translate-x-0'}
      `}>
        <div className="p-4 border-b border-bgBorder mt-12">
          <h3 className="text-xs font-semibold text-textMuted uppercase tracking-wider mb-1">Active Tools</h3>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto p-4 custom-scrollbar">
          <ToolStatus activeTool={activeTool} reasoningSteps={activeReasoning} />
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
