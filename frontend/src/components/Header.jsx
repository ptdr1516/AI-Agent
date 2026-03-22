import React, { useState } from 'react';

export default function Header({ title, isTyping }) {
  const [modelDropdown, setModelDropdown] = useState(false);

  return (
    <div className="h-14 flex-shrink-0 flex items-center justify-between px-6 border-b border-bgBorder bg-bgMain/90 backdrop-blur-md z-30 sticky top-0 shadow-sm">
      <div className="flex items-center gap-3">
        <h2 className="font-medium text-textMain text-[15px] truncate max-w-sm">{title}</h2>
        {isTyping && (
           <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-accentGlow border border-accentMain/20 text-accentMain text-xs font-medium animate-fade-in shadow-[0_0_8px_rgba(59,130,246,0.1)]">
             <span className="w-1.5 h-1.5 rounded-full bg-accentMain animate-pulse"></span>
             Computing
           </span>
        )}
      </div>

      <div className="flex items-center gap-3 relative">
        <button 
          onClick={() => setModelDropdown(!modelDropdown)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-bgHover border border-bgBorder text-sm text-textMuted hover:text-textMain hover:border-gray-600 transition-all select-none group"
        >
          <svg className="w-4 h-4 text-accentMain" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <span className="font-medium">gpt-3.5-turbo</span>
          <svg className={`w-3.5 h-3.5 transition-transform duration-200 ${modelDropdown ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {/* Mock Dropdown */}
        {modelDropdown && (
          <div className="absolute top-10 right-0 w-56 bg-bgPanel border border-bgBorder rounded-xl shadow-xl overflow-hidden animate-slide-up z-50">
            <div className="p-2 border-b border-bgBorder">
              <p className="text-xs font-semibold text-textFaint uppercase tracking-wider px-2 py-1">Models</p>
            </div>
            <div className="p-1">
              <button className="w-full text-left px-3 py-2 text-sm text-textMain bg-bgHover rounded-md flex justify-between items-center group">
                <span className="font-medium">gpt-3.5-turbo <span className="text-xs text-textFaint ml-1 bg-white/5 px-1 rounded">Fast</span></span>
                <svg className="w-4 h-4 text-accentMain" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                 </svg>
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-textMuted hover:bg-[#1f222d] hover:text-textMain rounded-md flex justify-between items-center transition-colors">
                <span>gpt-4o-mini <span className="text-xs text-textFaint ml-1">Smart</span></span>
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-textMuted hover:bg-[#1f222d] hover:text-textMain rounded-md flex justify-between items-center transition-colors">
                <span>claude-3-haiku</span>
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
