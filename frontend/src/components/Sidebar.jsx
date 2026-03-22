import React from 'react';

export default function Sidebar({ conversations, activeId, onSelect, onNew, onDelete }) {
  return (
    <div className="w-[260px] bg-bgSidebar h-full flex-shrink-0 flex flex-col border-r border-bgBorder text-sm text-textMuted select-none transition-all duration-300">
      
      {/* Brand Label */}
      <div className="flex items-center gap-2 px-5 py-6 mb-2">
        <div className="w-6 h-6 rounded-md bg-accentMain flex items-center justify-center shadow-[0_0_12px_rgba(59,130,246,0.5)]">
          <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <span className="font-semibold text-textMain tracking-wide text-base">Nova Platform</span>
      </div>

      {/* New Chat Button */}
      <div className="px-3 mb-4">
        <button
          onClick={onNew}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg bg-bgMain text-textMain border border-bgBorder hover:bg-bgHover hover:border-gray-700 transition-all duration-200 group shadow-sm"
        >
          <svg className="w-4 h-4 text-textMuted group-hover:text-accentMain transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span className="font-medium">New Chat</span>
          <span className="ml-auto flex items-center gap-1 opacity-50 text-[10px] font-mono">
            <kbd className="bg-bgHover px-1.5 py-0.5 rounded border border-bgBorder">⌘</kbd>
            <kbd className="bg-bgHover px-1.5 py-0.5 rounded border border-bgBorder">K</kbd>
          </span>
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto px-3 space-y-1 custom-scrollbar">
        <h3 className="px-3 text-xs font-semibold text-textFaint uppercase tracking-wider mb-2 mt-4">Recent</h3>
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`w-full text-left px-3 py-2.5 rounded-lg transition-all duration-200 flex items-center gap-2.5 group cursor-pointer ${
              c.id === activeId 
                ? 'bg-bgHover text-textMain font-medium' 
                : 'text-textMuted hover:bg-[#1f222d] hover:text-textMain'
            }`}
            onClick={() => onSelect(c.id)}
          >
            <svg 
              className={`w-4 h-4 shrink-0 transition-colors ${c.id === activeId ? 'text-accentMain' : 'text-textFaint'}`} 
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <span className="truncate flex-1">{c.title}</span>
            
            <button
              onClick={(e) => {
                e.stopPropagation(); // prevent triggering the row's onSelect
                onDelete(c.id);
              }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-black/20 text-textFaint hover:text-red-400 transition-all shrink-0 -mr-1"
              aria-label="Delete chat"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
            </button>
          </div>
        ))}
      </div>

      {/* User Profile Footer */}
      <div className="p-4 mt-auto border-t border-bgBorder bg-black/10">
        <div className="flex items-center gap-3 px-2 py-2 rounded-lg hover:bg-bgHover cursor-pointer transition-colors">
          <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-accentMain to-purple-500 flex items-center justify-center text-white font-semibold shadow-inner">
            D
          </div>
          <div className="flex flex-col">
            <span className="text-textMain font-medium text-sm">Dell User</span>
            <span className="text-[11px] text-textFaint">Premium Plan</span>
          </div>
          <div className="ml-auto text-textFaint">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
