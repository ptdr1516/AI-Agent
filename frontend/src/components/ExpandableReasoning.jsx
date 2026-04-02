import React, { useState } from 'react';

export default function ExpandableReasoning({ steps, ragContext }) {
  const [isOpen, setIsOpen] = useState(false);
  if (!steps || steps.length === 0) return null;
  
  // Map tool IDs to human-readable names
  const getToolName = (toolId) => {
    const toolNames = {
      'calculator': 'Calculator',
      'rag': 'Document Search',
      'search': 'Web Search',
      'sql': 'SQL Database',
      'api': 'User Lookups',
      'custom_api': 'Custom API'
    };
    return toolNames[toolId] || toolId.split('_').pop() || toolId;
  };
  
  return (
    <div className="mb-4">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs font-semibold text-textMuted hover:text-textMain transition-colors bg-bgPanel/40 hover:bg-bgPanel px-4 py-2.5 rounded-full border border-bgBorder select-none shadow-sm"
      >
        <div className={`w-2 h-2 rounded-full ${isOpen ? 'bg-accentMain animate-pulse' : 'bg-textFaint'}`} />
        <span>{isOpen ? 'Hide Reasoning' : `View Reasoning (${steps.length} steps)`}</span>
        <svg 
          className={`w-3.5 h-3.5 transition-transform duration-300 ${isOpen ? 'rotate-180' : ''}`} 
          fill="none" viewBox="0 0 24 24" stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="mt-3 space-y-3 animate-slide-down">
          {steps.map((step, idx) => (
            <div key={idx} className="bg-black/20 rounded-xl p-3 border border-bgBorder/50 font-mono text-[11px]">
              <div className="flex items-center justify-between mb-2">
                <span className="text-accentMain font-bold uppercase tracking-wider bg-accentMain/10 px-1.5 py-0.5 rounded">
                  {getToolName(step.tool)}
                </span>
                {!step.output && (
                  <span className="text-[9px] text-accentMain animate-pulse font-bold">RUNNING</span>
                )}
              </div>
              
              <div className="space-y-1.5 leading-relaxed overflow-hidden">
                <div className="flex gap-2">
                  <span className="text-textFaint shrink-0">IN:</span>
                  <span className="text-textMuted break-all">{step.input}</span>
                </div>
                {step.output && (
                  <div className="flex gap-2 pt-1 border-t border-white/5">
                    <span className="text-emerald-500/60 shrink-0">OUT:</span>
                    <span className="text-emerald-200/40 break-all line-clamp-3">
                      {step.output}
                    </span>
                  </div>
                )}
              </div>
            </div>
          ))}
          
          {ragContext && (
            <div className="p-3 bg-accentMain/5 border border-accentMain/20 rounded-xl">
              <div className="flex items-center gap-2 mb-1">
                <svg className="w-3.5 h-3.5 text-accentMain" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-[10px] font-bold text-accentMain uppercase tracking-widest italic">RAG Context Injected</span>
              </div>
              <p className="text-[10px] text-textMuted leading-tight opacity-70">
                The agent is answering using retrieved knowledge base passages.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
