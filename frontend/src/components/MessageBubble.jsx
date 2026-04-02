import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import ExpandableReasoning from "./ExpandableReasoning";

const MessageAction = ({ onClick, icon, label, variant = 'default', active = false }) => (
  <button
    onClick={onClick}
    title={label}
    className={`p-1.5 rounded-md transition-all flex items-center justify-center ${
      active 
        ? 'bg-accentMain text-white shadow-sm' 
        : variant === 'danger'
          ? 'hover:bg-red-500/20 text-textFaint hover:text-red-400'
          : 'hover:bg-bgHover text-textFaint hover:text-textMain'
    }`}
  >
    {icon}
  </button>
);

export default function MessageBubble({ message, onDelete, onRetry, onPin, onEdit }) {
  const isAgent = message.role === 'agent';
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formattedTime = message.timestamp 
    ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';

  return (
    <div className={`group relative flex flex-col mb-6 px-4 md:px-0 transition-all ${isAgent ? 'items-start' : 'items-end'}`}>
      
      {/* Message Toolbar - Hover state */}
      <div className={`absolute -top-3 ${isAgent ? 'left-4 md:left-2' : 'right-4 md:right-2'} z-20 flex items-center gap-1 bg-bgMain border border-bgBorder rounded-lg p-1 shadow-xl opacity-0 group-hover:opacity-100 transition-all translate-y-2 group-hover:translate-y-0`}>
        <MessageAction 
          label={copied ? "Copied!" : "Copy"} 
          icon={copied ? (
            <svg className="w-3.5 h-3.5 text-emerald-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" /></svg>
          ) : (
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" /></svg>
          )} 
          onClick={handleCopy} 
        />
        
        {isAgent && onRetry && (
          <MessageAction 
            label="Retry" 
            icon={<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>}
            onClick={onRetry}
          />
        )}

        <MessageAction 
          label={message.isPinned ? "Unpin" : "Pin Message"} 
          active={message.isPinned}
          icon={<svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20"><path d="M5.05 3a.5.5 0 0 1 .5-.5h8.9a.5.5 0 0 1 .5.5v2c0 .217-.14.409-.348.477L13 6.05v5.034l1.832 2.018a.5.5 0 0 1-.332.848H5.5a.5.5 0 0 1-.332-.848L7 13.084V8.05l-.652-.623A.5.5 0 0 1 6 7.05V3z" /></svg>}
          onClick={onPin}
        />

        <div className="w-px h-3 bg-bgBorder mx-0.5" />
        
        {!isAgent && (
          <>
            <MessageAction 
              label="Edit" 
              icon={<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>}
              onClick={onEdit}
            />
            <MessageAction 
              label="Delete" 
              variant="danger"
              icon={<svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" /></svg>}
              onClick={onDelete}
            />
          </>
        )}
      </div>

      <div className={`flex max-w-[85%] md:max-w-[75%] gap-3 ${isAgent ? 'flex-row' : 'flex-row-reverse'}`}>
        {/* Avatar */}
        <div className={`mt-1 flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center text-xs font-bold border ${
          isAgent ? 'bg-bgPanel border-bgBorder text-accentMain shadow-sm' : 'bg-accentMain border-blue-400 text-white shadow-[0_0_12px_rgba(59,130,246,0.5)]'
        }`}>
          {isAgent ? 'N' : 'U'}
        </div>

        {/* Content */}
        <div className="flex flex-col gap-1">
          <div className={`message-bubble relative p-4 rounded-2xl text-[14.5px] leading-relaxed shadow-sm transition-all border ${
            isAgent 
              ? 'bg-[#1e222d] text-textMain rounded-tl-none border-[#2d323f]' 
              : 'bg-accentMain text-white rounded-tr-none border-blue-400'
          } ${message.isPinned ? 'ring-1 ring-accentMain shadow-[0_0_15px_rgba(59,130,246,0.2)]' : ''}`}>
            
            {message.isPinned && (
              <div className="absolute -top-1.5 -right-1.5 bg-accentMain text-white p-1 rounded-full shadow-lg z-10 border border-bgMain">
                <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20"><path d="M5.05 3a.5.5 0 0 1 .5-.5h8.9a.5.5 0 0 1 .5.5v2c0 .217-.14.409-.348.477L13 6.05v5.034l1.832 2.018a.5.5 0 0 1-.332.848H5.5a.5.5 0 0 1-.332-.848L7 13.084V8.05l-.652-.623A.5.5 0 0 1 6 7.05V3z" /></svg>
              </div>
            )}

            <ReactMarkdown
              components={{
                p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
                code({ node, inline, className, children, ...props }) {
                  const match = /language-(\w+)/.exec(className || '');
                  return !inline && match ? (
                    <div className="my-4 rounded-lg overflow-hidden border border-bgBorder shadow-lg">
                      <div className="bg-bgMenu px-4 py-2 text-[11px] font-mono text-textFaint border-b border-bgBorder flex justify-between items-center capitalize">
                        <span>{match[1]}</span>
                      </div>
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        className="!m-0 !bg-[#0d1117]"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    </div>
                  ) : (
                    <code className="bg-black/30 px-1.5 py-0.5 rounded text-[13px] font-mono" {...props}>
                      {children}
                    </code>
                  );
                },
              }}
            >
              {message.content}
            </ReactMarkdown>

            {isAgent && message.reasoningSteps && message.reasoningSteps.length > 0 && (
              <div className="mt-4 pt-3 border-t border-[#2d323f]">
                <ExpandableReasoning steps={message.reasoningSteps} />
              </div>
            )}
          </div>
          
          {/* Timestamp */}
          <span className={`text-[10px] text-textFaint uppercase tracking-wider mt-1 px-1 ${isAgent ? 'text-left' : 'text-right'}`}>
            {formattedTime}
          </span>
        </div>
      </div>
    </div>
  );
}
