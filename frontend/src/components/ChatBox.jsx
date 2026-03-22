import React, { useEffect, useRef } from 'react';

const ChatBox = ({ messages, isTyping }) => {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-6">
      {messages.map((msg, idx) => (
        <div 
          key={idx} 
          className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'} message-appear`}
        >
          <div className={`max-w-[80%] rounded-2xl p-4 shadow-lg ${
            msg.role === 'user' 
              ? 'bg-primary text-white rounded-tr-sm' 
              : 'bg-cardBg text-gray-200 border border-gray-700/50 rounded-tl-sm'
          }`}>
            <p className="whitespace-pre-wrap leading-relaxed">{msg.content}</p>
          </div>
        </div>
      ))}
      
      {isTyping && (
        <div className="flex w-full justify-start message-appear">
          <div className="max-w-[80%] rounded-2xl p-4 bg-cardBg border border-gray-700/50 rounded-tl-sm flex items-center space-x-2 shadow-lg">
            <div className="w-2 h-2 rounded-full bg-primary animate-bounce"></div>
            <div className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            <div className="w-2 h-2 rounded-full bg-primary animate-bounce" style={{ animationDelay: '0.4s' }}></div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
};

export default ChatBox;
