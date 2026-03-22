import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript';
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql';
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash';
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json';

SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('javascript', javascript);
SyntaxHighlighter.registerLanguage('sql', sql);
SyntaxHighlighter.registerLanguage('bash', bash);
SyntaxHighlighter.registerLanguage('json', json);

function RagContextPanel({ ragContext }) {
  if (!ragContext?.chunks?.length) return null;
  const { sources = [], chunks } = ragContext;
  return (
    <div className="mt-6 rounded-xl border border-bgBorder overflow-hidden bg-black/25 shadow-inner">
      <div className="px-4 py-3 border-b border-bgBorder bg-black/30">
        <h4 className="text-[11px] font-semibold text-textMuted uppercase tracking-wider mb-2">Sources</h4>
        <div className="flex flex-wrap gap-2">
          {sources.map((name) => (
            <span
              key={name}
              className="text-xs px-2.5 py-1 rounded-md bg-accentMain/15 text-accentMain font-mono border border-accentMain/20"
            >
              {name}
            </span>
          ))}
        </div>
      </div>
      <div className="px-4 py-3">
        <h4 className="text-[11px] font-semibold text-textMuted uppercase tracking-wider mb-3">Chunk previews</h4>
        <ul className="space-y-3 max-h-[min(24rem,50vh)] overflow-y-auto pr-1">
          {chunks.map((c, idx) => (
            <li
              key={`${c.filename}-${idx}`}
              className="rounded-lg border border-bgBorder/70 bg-[#0d1017]/90 p-3"
            >
              <div className="text-xs font-semibold text-accentMain mb-2 truncate" title={c.filename}>
                {c.filename}
              </div>
              <pre className="text-[13px] text-textMuted/95 whitespace-pre-wrap break-words font-sans leading-relaxed line-clamp-6">
                {c.preview}
              </pre>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

const ExpandableReasoning = ({ steps }) => {
  const [isOpen, setIsOpen] = useState(false);
  
  if (!steps || steps.length === 0) return null;
  
  return (
    <div className="mb-4">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs font-semibold text-textMuted hover:text-textMain transition-colors bg-black/20 hover:bg-black/40 px-3 py-1.5 rounded-full border border-bgBorder select-none"
      >
        <svg className={`w-3.5 h-3.5 transition-transform duration-200 ${isOpen ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
        </svg>
        {isOpen ? "Hide Reasoning" : "View Reasoning"}
        {steps.some(s => !s.output) && (
           <span className="flex items-center gap-1.5 ml-1 px-1.5 bg-accentGlow text-accentMain rounded-sm">
             <span className="w-1.5 h-1.5 rounded-full bg-accentMain animate-pulse"></span>
             <span className="text-[10px] tracking-wider uppercase">Running</span>
           </span>
        )}
      </button>
      
      {isOpen && (
        <div className="mt-3 space-y-4 bg-[#0d1017] border border-bgBorder rounded-xl p-4 shadow-inner animate-fade-in text-[13px] font-mono">
          {steps.map((step, idx) => (
            <div key={idx} className="border-l-2 border-bgBorder pl-4 py-0.5 hover:border-accentMain/40 transition-colors">
              <div className="flex items-center gap-2 mb-1.5">
                <span className="text-accentMain font-semibold uppercase tracking-wider text-[11px] bg-accentMain/10 px-2 py-0.5 rounded">{step.tool}</span>
              </div>
              
              <div className="mb-1 grid grid-cols-[50px_1fr] gap-2 items-start">
                <span className="text-textFaint select-none">Input:</span>
                <span className="text-textMuted break-all whitespace-pre-wrap">{step.input}</span>
              </div>
              
              {step.output ? (
                <div className="grid grid-cols-[50px_1fr] gap-2 items-start">
                  <span className="text-textFaint select-none">Output:</span>
                  <span className="text-green-400/80 break-all whitespace-pre-wrap">{step.output.length > 500 ? step.output.slice(0, 500) + '... (truncated)' : step.output}</span>
                </div>
              ) : (
                <div className="mt-2 flex items-center gap-2">
                   <div className="typing-dots"><span/><span/><span/></div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default function MessageBubble({ message }) {
  const isAgent = message.role === 'agent';

  return (
    <div className={`flex w-full ${isAgent ? 'justify-start' : 'justify-end'} mb-6 group animate-slide-up`}>
      {isAgent && (
        <div className="w-8 h-8 rounded-full bg-accentMain flex items-center justify-center shrink-0 shadow-[0_0_10px_rgba(59,130,246,0.3)] mt-0.5 mr-4 overflow-hidden border border-accentMain/50">
          <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
      )}

      <div className={`max-w-[85%] ${isAgent ? 'text-textMain prose prose-invert w-full min-w-0' : 'bg-bgHover px-5 py-3 rounded-2xl rounded-tr-sm text-textMain'}`}>
        {isAgent ? (
          <>
            <ExpandableReasoning steps={message.reasoningSteps} />
            {message.ragContext?.chunks?.length > 0 && (
              <p className="text-[11px] font-semibold text-textMuted uppercase tracking-wider mb-2">Answer</p>
            )}
            <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              code({ node, inline, className, children, ...props }) {
                const match = /language-(\w+)/.exec(className || '');
                const [copied, setCopied] = useState(false);

                const handleCopy = () => {
                  navigator.clipboard.writeText(String(children).replace(/\n$/, ''));
                  setCopied(true);
                  setTimeout(() => setCopied(false), 2000);
                };

                return !inline && match ? (
                  <div className="relative group/code rounded-lg overflow-hidden border border-bgBorder my-4 bg-[#0d1017]">
                    <div className="flex items-center justify-between px-4 py-1.5 bg-black/40 border-b border-bgBorder text-xs text-textMuted font-mono">
                      <span>{match[1]}</span>
                      <button 
                        onClick={handleCopy}
                        className="hover:text-textMain transition-colors flex items-center gap-1.5"
                      >
                        {copied ? (
                           <><svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg> Copied</>
                        ) : (
                           <><svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg> Copy</>
                        )}
                      </button>
                    </div>
                    <SyntaxHighlighter
                      style={vscDarkPlus}
                      language={match[1]}
                      PreTag="div"
                      customStyle={{ margin: 0, padding: '1rem', background: 'transparent' }}
                      {...props}
                    >
                      {String(children).replace(/\n$/, '')}
                    </SyntaxHighlighter>
                  </div>
                ) : (
                  <code className="bg-black/30 rounded px-1.5 py-0.5 text-sm font-mono text-pink-300" {...props}>
                    {children}
                  </code>
                );
              }
            }}
          >
            {message.content}
          </ReactMarkdown>
            <RagContextPanel ragContext={message.ragContext} />
          </>
        ) : (
          <div className="whitespace-pre-wrap leading-relaxed">{message.content}</div>
        )}
      </div>
    </div>
  );
}
