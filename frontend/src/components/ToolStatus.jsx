import React from 'react';

const tools = [
  {
    id: 'calculator',
    name: 'Calculator',
    description: 'Evaluates math expressions safely',
    icon: (
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
      </svg>
    )
  },
  {
    id: 'rag',
    name: 'Document search',
    description: 'Retrieves passages from uploaded documents',
    icon: (
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    )
  },
  {
    id: 'search',
    name: 'Web Search',
    description: 'Searches internet for live data',
    icon: (
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9" />
      </svg>
    )
  },
  {
    id: 'sql',
    name: 'SQL Database',
    description: 'Queries internal HR database',
    icon: (
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
      </svg>
    )
  },
  {
    id: 'api',
    name: 'User Lookups',
    description: 'Mock internal API for users',
    icon: (
      <svg fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
      </svg>
    )
  }
];

export default function ToolStatus({ activeTool, reasoningSteps = [] }) {
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
    <div className="space-y-6 flex flex-col h-full">
      {/* Active Tools Section */}
      <section>
        <h4 className="text-[10px] font-bold text-textMuted uppercase tracking-widest mb-3 ml-1 opacity-70">Infrastructure</h4>
        <div className="space-y-2">
          {tools.map((tool) => {
            const isActive = activeTool === tool.id;
            return (
              <div 
                key={tool.id} 
                className={`flex items-center gap-3 px-3 py-2 rounded-lg border transition-all duration-200 ${
                  isActive 
                    ? 'bg-accentMain/10 border-accentMain/40 ring-1 ring-accentMain/20' 
                    : 'bg-bgMain/40 border-bgBorder hover:border-gray-700'
                }`}
              >
                <div className={`w-7 h-7 rounded-md flex items-center justify-center shrink-0 transition-all ${
                  isActive ? 'bg-accentMain text-white shadow-[0_0_8px_rgba(59,130,246,0.3)]' : 'bg-black/20 text-textFaint'
                }`}>
                  <div className="w-3.5 h-3.5">{tool.icon}</div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-2">
                    <span className={`text-[13px] font-medium truncate ${isActive ? 'text-textMain' : 'text-textMuted'}`}>
                      {tool.name}
                    </span>
                    {isActive && (
                      <span className="w-1.5 h-1.5 rounded-full bg-accentMain animate-pulse shadow-[0_0_5px_rgba(59,130,246,0.8)]" />
                    )}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Execution Log Section */}
      <section className="flex-1 flex flex-col min-h-0">
        <h4 className="text-[10px] font-bold text-textMuted uppercase tracking-widest mb-3 ml-1 opacity-70 flex justify-between items-center">
          Execution Log
          {reasoningSteps.length > 0 && <span className="text-accentMain lowercase normal-case font-mono">{reasoningSteps.length} items</span>}
        </h4>
        
        <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-3 custom-scrollbar">
          {reasoningSteps.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center p-4 border border-dashed border-bgBorder rounded-xl bg-black/10">
              <svg className="w-8 h-8 text-textFaint mb-2 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-[11px] text-textFaint uppercase tracking-tighter">Waiting for tools...</p>
            </div>
          ) : (
            <div className="space-y-4 font-mono">
              {reasoningSteps.map((step, idx) => (
                <div key={idx} className="group animate-fade-in relative pl-4 border-l border-bgBorder hover:border-accentMain/40 transition-colors">
                  <div className="absolute left-[-5px] top-1.5 w-2 h-2 rounded-full bg-bgBorder group-hover:bg-accentMain transition-colors" />
                  
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] font-bold text-accentMain uppercase tracking-wider bg-accentMain/10 px-1.5 py-0.5 rounded">
                      {getToolName(step.tool)}
                    </span>
                    {!step.output && (
                      <span className="text-[9px] text-accentMain animate-pulse font-bold tracking-tighter">RUNNING</span>
                    )}
                  </div>

                  <div className="space-y-2 text-[11px] leading-relaxed">
                    <div className="bg-black/25 rounded-md p-2 border border-bgBorder/50">
                      <span className="text-textFaint mr-2">QUERY:</span>
                      <span className="text-textMuted break-all">{step.input}</span>
                    </div>
                    
                    {step.output && (
                      <div className="bg-emerald-500/5 rounded-md p-2 border border-emerald-500/10">
                        <span className="text-emerald-500/60 mr-2 uppercase text-[9px] font-bold tracking-widest">DATA:</span>
                        <span className="text-emerald-200/40 break-all truncate block">
                          {step.output.length > 100 ? step.output.slice(0, 100) + '...' : step.output}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
