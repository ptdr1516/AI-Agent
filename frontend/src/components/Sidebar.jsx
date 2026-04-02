import React, { useState, useMemo } from 'react';

const ChatOptionButton = ({ onClick, icon, label, variant = 'default' }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center gap-2 px-3 py-2 text-xs rounded-md transition-colors ${
      variant === 'danger' 
        ? 'text-red-400 hover:bg-red-500/10' 
        : 'text-textMuted hover:bg-bgHover hover:text-textMain'
    }`}
  >
    {icon}
    <span>{label}</span>
  </button>
);

export default function Sidebar({ conversations, activeId, onSelect, onNew, onDelete, onRename, onPin, onArchive, isOpen, onClose }) {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeMenuId, setActiveMenuId] = useState(null);
  const [showArchived, setShowArchived] = useState(false);
  const [editingId, setEditingId] = useState(null);
  const [editingTitle, setEditingTitle] = useState('');

  const filteredConversations = useMemo(() => {
    let list = conversations.filter(c => showArchived ? c.isArchived : !c.isArchived);
    
    if (searchQuery.trim()) {
      const lower = searchQuery.toLowerCase();
      list = list.filter(c => c.title.toLowerCase().includes(lower));
    }
    
    return list;
  }, [conversations, searchQuery, showArchived]);

  const pinnedConversations = useMemo(() => {
    return filteredConversations
      .filter(c => c.isPinned)
      .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
  }, [filteredConversations]);

  const recentConversations = useMemo(() => {
    return filteredConversations
      .filter(c => !c.isPinned)
      .sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
  }, [filteredConversations]);

  const handleRename = (id, currentTitle) => {
    setEditingId(id);
    setEditingTitle(currentTitle);
    setActiveMenuId(null);
  };

  const submitRename = () => {
    if (editingId && editingTitle.trim()) {
      onRename(editingId, editingTitle.trim());
    }
    setEditingId(null);
  };

  const ChatItem = ({ c }) => {
    const isEditing = editingId === c.id;
    
    return (
      <div 
        key={c.id} 
        className={`relative group animate-fade-in ${activeMenuId === c.id ? 'z-30' : 'z-auto'}`}
      >
      <div
        className={`w-full text-left px-3 py-2.5 rounded-lg transition-all duration-200 flex items-center gap-2.5 cursor-pointer ${
          c.id === activeId 
            ? 'bg-bgHover text-textMain font-medium shadow-sm' 
            : 'text-textMuted hover:bg-[#1f222d] hover:text-textMain'
        }`}
        onClick={() => onSelect(c.id)}
      >
        <div className="relative shrink-0">
          <svg className={`w-4 h-4 transition-colors ${c.id === activeId ? 'text-accentMain' : 'text-textFaint'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
          {c.isPinned && (
            <div className="absolute -top-1 -right-1 bg-accentMain text-white rounded-full p-0.5 shadow-sm">
              <svg className="w-1.5 h-1.5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M5.05 3a.5.5 0 0 1 .5-.5h8.9a.5.5 0 0 1 .5.5v2c0 .217-.14.409-.348.477L13 6.05v5.034l1.832 2.018a.5.5 0 0 1-.332.848H5.5a.5.5 0 0 1-.332-.848L7 13.084V8.05l-.652-.623A.5.5 0 0 1 6 7.05V3z" />
              </svg>
            </div>
          )}
        </div>
          
          {isEditing ? (
            <input
              autoFocus
              value={editingTitle}
              onChange={(e) => setEditingTitle(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') submitRename();
                if (e.key === 'Escape') setEditingId(null);
              }}
              onBlur={submitRename}
              className="flex-1 bg-black/40 border border-accentMain/50 rounded px-1.5 py-0.5 text-[13px] text-textMain outline-none focus:ring-1 focus:ring-accentMain/30 min-w-0"
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span className="truncate flex-1 text-[13px]">{c.title}</span>
          )}
          
          {!isEditing && (
            <button
          onClick={(e) => {
            e.stopPropagation();
            setActiveMenuId(activeMenuId === c.id ? null : c.id);
          }}
          className={`p-1 rounded-md hover:bg-black/20 text-textFaint hover:text-textMain transition-all opacity-0 group-hover:opacity-100 ${activeMenuId === c.id ? 'opacity-100 bg-black/20' : ''}`}
        >
          <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
          </svg>
        </button>
      )}
    </div>

      {/* Dropdown Menu */}
      {activeMenuId === c.id && (
        <React.Fragment>
          <div className="fixed inset-0 z-10" onClick={() => setActiveMenuId(null)} />
          <div className="absolute right-2 top-10 w-40 bg-[#161922] border border-bgBorder rounded-xl shadow-[0_10px_40px_rgba(0,0,0,0.7)] z-50 py-1.5 overflow-hidden animate-slide-up">
            <ChatOptionButton 
              label="Rename" 
              icon={<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>}
              onClick={() => handleRename(c.id, c.title)}
            />
            <ChatOptionButton 
              label={c.isPinned ? "Unpin Chat" : "Pin Chat"} 
              icon={<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" /></svg>}
              onClick={() => { onPin(c.id); setActiveMenuId(null); }}
            />
            <ChatOptionButton 
              label={c.isArchived ? "Unarchive" : "Archive Chat"} 
              icon={<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" /></svg>}
              onClick={() => { onArchive(c.id); setActiveMenuId(null); }}
            />
            <div className="h-px bg-bgBorder my-1" />
            <ChatOptionButton 
              label="Delete Chat" 
              variant="danger"
              icon={<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>}
              onClick={() => { onDelete(c.id); setActiveMenuId(null); }}
            />
          </div>
        </React.Fragment>
      )}
    </div>
    );
  };

  return (
    <div className={`
      fixed inset-y-0 left-0 z-50 md:relative md:z-auto h-full flex-shrink-0 flex flex-col border-r border-bgBorder bg-bgSidebar text-sm text-textMuted select-none transition-all duration-300 ease-in-out
      ${isOpen ? 'translate-x-0 w-4/5 sm:w-72 md:w-64 xl:w-72 opacity-100 shadow-2xl md:shadow-none' : '-translate-x-full md:translate-x-0 md:w-0 md:opacity-0 md:border-none overflow-hidden'}
    `}>
      {/* Brand Label */}
      <div className="flex items-center gap-2 px-5 py-6 mb-2">
        <div className="w-6 h-6 rounded-md bg-accentMain flex items-center justify-center shadow-[0_0_12px_rgba(59,130,246,0.5)]">
          <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <span className="font-semibold text-textMain tracking-wide text-base">Nova Platform</span>
      </div>

      {/* Sidebar Search */}
      <div className="px-3 mb-3">
        <div className="relative group">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search conversations..."
            className="w-full bg-bgMain border border-bgBorder rounded-lg py-2 pl-9 pr-8 text-xs text-textMain placeholder:text-textFaint focus:border-accentMain/40 focus:ring-1 focus:ring-accentMain/10 outline-none transition-all"
          />
          <div className="absolute inset-y-0 left-3 flex items-center pointer-events-none">
            <svg className="w-3.5 h-3.5 text-textFaint" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>
          {searchQuery && (
            <button onClick={() => setSearchQuery('')} className="absolute inset-y-0 right-2 flex items-center px-1 text-textFaint hover:text-textMain">
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
          )}
        </div>
      </div>

      {/* New Chat Button */}
      <div className="px-3 mb-4">
        <button
          onClick={onNew}
          className="w-full flex items-center gap-2.5 px-3 py-3 rounded-lg bg-bgMain text-textMain border border-bgBorder hover:bg-bgHover hover:border-gray-700 transition-all duration-200 group shadow-sm"
        >
          <svg className="w-4 h-4 text-textMuted group-hover:text-accentMain transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span className="font-medium">New Chat</span>
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto px-3 space-y-1 custom-scrollbar">
        <div className="flex items-center justify-between px-3 mb-2 mt-4">
          <h3 className="text-[10px] font-bold text-textFaint uppercase tracking-[0.1em]">
            {showArchived ? 'Archived' : 'Conversations'}
          </h3>
          <button 
            onClick={() => setShowArchived(!showArchived)}
            className={`text-[9px] font-black px-1.5 py-0.5 rounded transition-all ${showArchived ? 'bg-accentMain text-white' : 'bg-bgHover text-textFaint hover:text-textMain'}`}
          >
            {showArchived ? 'DONE' : 'VIEW ARCHIVE'}
          </button>
        </div>

        {filteredConversations.length === 0 && (
           <div className="px-3 py-8 text-center italic text-xs text-textFaint opacity-40">
             {searchQuery ? 'No results found' : 'No chats here yet'}
           </div>
        )}

        {/* Pinned Section */}
        {!showArchived && pinnedConversations.length > 0 && (
          <div className="mb-4">
            <div className="px-3 py-1 text-[9px] font-bold text-accentMain uppercase tracking-widest opacity-60">Pinned</div>
            {pinnedConversations.map(c => <ChatItem key={c.id} c={c} />)}
          </div>
        )}

        {/* Recent Section */}
        {recentConversations.length > 0 && (
          <div>
            {!showArchived && pinnedConversations.length > 0 && (
              <div className="px-3 py-1 text-[9px] font-bold text-textFaint uppercase tracking-widest opacity-40 mt-2">Recent</div>
            )}
            {recentConversations.map(c => <ChatItem key={c.id} c={c} />)}
          </div>
        )}
      </div>

      {/* User Profile Footer */}
      <div className="p-4 mt-auto border-t border-bgBorder bg-black/10">
        <div className="flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-bgHover cursor-pointer transition-colors">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accentMain to-purple-500 flex items-center justify-center text-white font-semibold shadow-inner">
            D
          </div>
          <div className="flex flex-col">
            <span className="text-textMain font-medium text-sm">Dell User</span>
            <span className="text-[11px] text-textFaint uppercase tracking-tighter">Premium Agent</span>
          </div>
        </div>
      </div>
    </div>
  );
}
