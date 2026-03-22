import React, { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';

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
  <div className="h-full flex flex-col items-center justify-center max-w-2xl mx-auto px-6 text-center animate-fade-in">
    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accentMain to-purple-600 flex items-center justify-center shadow-xl mb-6">
      <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    </div>
    <h2 className="text-2xl font-semibold text-textMain mb-3">How can I help?</h2>
    <p className="text-textMuted mb-10 max-w-md leading-relaxed">
      Ask me anything — I can calculate math, search the web, query the employee database, or look up users.
    </p>
    
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-lg">
      {[
        'What is 1234 × 56?', 
        'Who is user 123?', 
        'List all employees', 
        'Tell me about quantum computing'
      ].map(p => (
        <div key={p} className="px-4 py-3 rounded-xl border border-bgBorder bg-bgHover/50 text-sm text-textMuted hover:text-textMain hover:border-gray-600 hover:bg-bgHover cursor-pointer transition-all duration-200 text-left">
          {p}
        </div>
      ))}
    </div>
  </div>
);

export default function ChatWindow({ messages, isTyping }) {
  const bottomRef = useRef(null);
  
  useEffect(() => { 
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isTyping]);

  return (
    <div className="h-full w-full flex flex-col">
      {messages.length === 0 ? (
        <EmptyState />
      ) : (
        <div className="flex-1 w-full max-w-3xl mx-auto px-4 py-8">
          {messages.map((msg, idx) => <MessageBubble key={idx} message={msg} />)}
          
          {/* Only show typing if waiting for agent to respond (i.e. last message is user) */}
          {isTyping && messages[messages.length - 1]?.role === 'user' && <TypingIndicator />}
          
          <div ref={bottomRef} className="h-2" />
        </div>
      )}
    </div>
  );
}
