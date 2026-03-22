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

export default function ToolStatus({ activeTool }) {
  return (
    <div className="space-y-3">
      {tools.map((tool) => {
        const isActive = activeTool === tool.id;
        
        return (
          <div 
            key={tool.id} 
            className={`flex items-start gap-3 p-3 rounded-xl border transition-all duration-300 ${
              isActive 
                ? 'bg-accentGlow border-accentMain/30 shadow-[0_0_15px_rgba(59,130,246,0.1)]' 
                : 'bg-bgHover border-bgBorder hover:border-gray-600'
            }`}
          >
            <div className={`mt-0.5 w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
              isActive ? 'bg-accentMain text-white' : 'bg-bgMain text-textMuted'
            }`}>
              <div className="w-4 h-4">{tool.icon}</div>
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex justify-between items-center mb-0.5">
                <span className={`text-sm font-semibold truncate ${isActive ? 'text-accentMain' : 'text-textMain'}`}>
                  {tool.name}
                </span>
                {isActive ? (
                  <span className="flex items-center gap-1.5 px-2 py-0.5 bg-accentMain text-white text-[10px] font-bold uppercase rounded-sm shadow-sm animate-pulse-fast tracking-wider">
                    Running
                  </span>
                ) : (
                  <span className="hidden group-hover:block px-2 py-0.5 bg-bgMain text-textFaint text-[10px] font-medium uppercase rounded shadow-sm border border-bgBorder">
                    Ready
                  </span>
                )}
              </div>
              <p className="text-xs text-textMuted line-clamp-2 leading-relaxed">{tool.description}</p>
            </div>
          </div>
        );
      })}
    </div>
  );
}
