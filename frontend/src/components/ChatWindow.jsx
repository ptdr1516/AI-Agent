import React, { useEffect, useRef, useState, useMemo } from 'react';
import MessageBubble from './MessageBubble';

const DateSeparator = ({ date }) => (
  <div className="flex items-center gap-4 my-8 opacity-40 select-none">
    <div className="flex-1 h-px bg-gradient-to-r from-transparent to-bgBorder" />
    <span className="text-[10px] font-bold tracking-[0.2em] text-textMuted uppercase whitespace-nowrap">
      {date}
    </span>
    <div className="flex-1 h-px bg-gradient-to-l from-transparent to-bgBorder" />
  </div>
);

const TypingIndicator = () => (
  <div className="flex animate-fade-in mb-6">
    <div className="w-8 h-8 rounded-full bg-accentMain flex items-center justify-center shrink-0 shadow-[0_0_10px_rgba(59,130,246,0.3)] mt-0.5">
      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    </div>
    <div className="ml-4 flex items-center h-9">
      <div className="typing-dots flex items-center h-full">
        <span />
        <span />
        <span />
      </div>
    </div>
  </div>
);

const EmptyState = () => (
  <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto px-6 text-center animate-fade-in py-12">
    <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-accentMain via-blue-600 to-purple-600 flex items-center justify-center shadow-2xl mb-8 transform hover:scale-105 transition-transform duration-300">
      <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    </div>
    <h2 className="text-3xl font-bold text-textMain mb-4 tracking-tight">How can I help today?</h2>
    <p className="text-textMuted mb-12 max-w-md leading-relaxed text-lg opacity-80">
      Nova Agent is ready to calculate, search, and analyze documents with lightning speed.
    </p>
    
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-xl">
      {[
        { t: 'What is 1234 × 56?', i: '🔢' },
        { t: 'Who is user 123?', i: '👤' },
        { t: 'List all employees', i: '📋' },
        { t: 'Tell me about quantum computing', i: '🌌' }
      ].map(p => (
        <div key={p.t} className="group relative px-5 py-4 rounded-2xl border border-bgBorder bg-bgPanel/40 hover:bg-bgHover hover:border-accentMain/40 cursor-pointer transition-all duration-300 text-left overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-accentMain/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
          <span className="text-xl mb-2 block">{p.i}</span>
          <span className="text-sm font-medium text-textMuted group-hover:text-textMain transition-colors relative z-10">{p.t}</span>
        </div>
      ))}
    </div>
  </div>
);

const formatDate = (timestamp) => {
  if (!timestamp) return null;
  const d = new Date(timestamp);
  const now = new Date();
  
  if (d.toDateString() === now.toDateString()) return 'TODAY';
  
  const yesterday = new Date();
  yesterday.setDate(now.getDate() - 1);
  if (d.toDateString() === yesterday.toDateString()) return 'YESTERDAY';
  
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }).toUpperCase();
};

export default function ChatWindow({ messages, isTyping, onDelete, onRetry, onPinMessage, searchTerm = '', onEditMessage }) {
  const bottomRef = useRef(null);
  const scrollRef = useRef(null);
  const [showScrollBottom, setShowScrollBottom] = useState(false);

  // Group messages by date
  const groupedMessages = useMemo(() => {
    const groups = [];
    let lastDate = null;
    
    messages.forEach((msg) => {
      const dateStr = formatDate(msg.timestamp);
      if (dateStr !== lastDate) {
        groups.push({ type: 'date', value: dateStr, id: `date-${msg.id}` });
        lastDate = dateStr;
      }
      groups.push({ type: 'message', value: msg, id: msg.id });
    });
    
    return groups;
  }, [messages]);

  const handleScroll = (e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.target;
    // Show button if we are more than 300px away from bottom
    const isNearBottom = scrollHeight - scrollTop - clientHeight < 300;
    setShowScrollBottom(!isNearBottom);
  };

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => { 
    if (!showScrollBottom) {
      bottomRef.current?.scrollIntoView({ behavior: 'auto' });
    }
  }, [messages, isTyping, showScrollBottom]);

  return (
    <div 
      ref={scrollRef}
      onScroll={handleScroll}
      className="w-full flex-1 flex flex-col overflow-y-auto relative custom-scrollbar scroll-smooth"
    >
      {messages.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="flex-1 w-full max-w-3xl mx-auto px-4 py-8 relative">
          {groupedMessages.map((item) => (
            item.type === 'date' 
              ? <DateSeparator key={item.id} date={item.value} />
              : <MessageBubble 
                  key={item.id} 
                  message={item.value} 
                  onDelete={() => onDelete(item.value.id)}
                  onRetry={onRetry}
                  onPin={() => onPinMessage(item.value.id)}
                  onEdit={() => onEditMessage && onEditMessage(item.value)}
                />
          ))}
          
          {isTyping && (
            messages[messages.length - 1]?.role === 'user' || 
            (messages[messages.length - 1]?.role === 'agent' && !messages[messages.length - 1]?.content)
          ) && <TypingIndicator />}
          
          <div ref={bottomRef} className="h-4" />
        </div>
      )}

      {/* Floating Scroll to Bottom Button */}
      {showScrollBottom && (
        <button
          onClick={scrollToBottom}
          className="fixed bottom-32 left-1/2 -translate-x-1/2 z-40 bg-accentMain text-white rounded-full p-3 shadow-2xl hover:bg-blue-600 transition-all active:scale-95 flex items-center gap-2 group border border-white/20"
        >
          <svg className="w-5 h-5 animate-bounce" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M19 14l-7 7-7-7" />
          </svg>
          <span className="text-xs font-bold uppercase tracking-wider pr-1">New Messages</span>
        </button>
      )}
    </div>
  );
}
