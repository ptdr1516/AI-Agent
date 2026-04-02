import React, { useState, useRef, useEffect } from 'react';

export default function Header({ activeTitle, isTyping, onMenuClick, onToggleSidebar, onToolsClick, searchTerm, onSearchChange, isOpen }) {
  const [modelDropdown, setModelDropdown] = useState(false);
  const [isSearchOpen, setIsSearchOpen] = useState(false);
  const searchInputRef = useRef(null);

  useEffect(() => {
    if (isSearchOpen && searchInputRef.current) {
      searchInputRef.current.focus();
    }
  }, [isSearchOpen]);

  const handleToggleSearch = () => {
    if (isSearchOpen) {
      onSearchChange(''); // Clear on close
    }
    setIsSearchOpen(!isSearchOpen);
  };

  return (
    <div className="h-14 flex-shrink-0 flex items-center justify-between px-2 sm:px-4 md:px-6 border-b border-bgBorder bg-bgMain/90 backdrop-blur-md z-30 sticky top-0 shadow-sm gap-1.5">
      <div className="flex items-center gap-2 sm:gap-3 flex-1 min-w-0">
        <button 
          onClick={onMenuClick} 
          className="md:hidden p-3 -ml-2.5 flex-shrink-0 text-textMuted hover:text-textMain rounded-md hover:bg-bgHover transition-colors focus:outline-none focus:ring-2 focus:ring-accentMain/50"
          aria-label="Toggle Sidebar"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>

        {/* Global Sidebar Toggle (Visible everywhere except ultra-mobile where Menu takes over) */}
        <button 
          onClick={onToggleSidebar}
          className="hidden md:flex p-2.5 text-textMuted hover:text-textMain rounded-lg hover:bg-bgHover transition-colors focus:outline-none focus:ring-2 focus:ring-accentMain/50 mr-1"
          title={isSearchOpen ? "Close Search first" : "Toggle Sidebar (Ctrl+\\)"}
          disabled={isSearchOpen}
        >
          <svg className={`w-5 h-5 transition-transform duration-300 ${!isOpen ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
          </svg>
        </button>

        {isSearchOpen ? (
          <div className="flex-1 flex items-center bg-black/20 rounded-lg px-3 py-1.5 border border-bgBorder animate-fade-in">
            <svg className="w-4 h-4 text-textFaint mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              ref={searchInputRef}
              type="text"
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              placeholder="Search conversation..."
              className="bg-transparent border-none outline-none text-sm text-textMain w-full placeholder:text-textFaint"
            />
            <button 
              onClick={handleToggleSearch}
              className="ml-2 text-textFaint hover:text-textMain"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2 min-w-0 flex-1">
            <h2 className="font-medium text-textMain text-[15px] truncate">{activeTitle}</h2>
            {isTyping && (
               <span className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-accentGlow border border-accentMain/20 text-accentMain text-xs font-medium animate-fade-in flex-shrink-0">
                 <span className="w-1.5 h-1.5 rounded-full bg-accentMain animate-pulse flex-shrink-0"></span>
                 Thinking...
               </span>
            )}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0 relative">
        {!isSearchOpen && (
          <button 
            onClick={handleToggleSearch}
            className="p-3 text-textMuted hover:text-textMain rounded-md hover:bg-bgHover transition-colors focus:outline-none focus:ring-2 focus:ring-accentMain/50"
            aria-label="Search conversation"
            title="Search"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
        )}

        <button 
          onClick={onToolsClick}
          className="xl:hidden p-3 -mr-1 text-textMuted hover:text-textMain rounded-md hover:bg-bgHover transition-colors focus:outline-none focus:ring-2 focus:ring-accentMain/50"
          aria-label="Toggle Tools Panel"
          title="Active Tools"
        >
          <svg className="w-5 h-5 text-accentMain/80 hover:text-accentMain transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
          </svg>
        </button>

        <button 
          onClick={() => setModelDropdown(!modelDropdown)}
          className="hidden sm:flex items-center gap-2 px-3 py-2 rounded-lg bg-bgHover border border-bgBorder text-sm text-textMuted hover:text-textMain hover:border-gray-600 transition-all select-none group"
        >
          <svg className="w-4 h-4 text-accentMain" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <span className="font-medium">gpt-3.5-turbo</span>
          <svg className={`w-3.5 h-3.5 transition-transform duration-200 ${modelDropdown ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

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
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
