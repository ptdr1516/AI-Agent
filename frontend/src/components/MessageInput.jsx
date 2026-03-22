import React, { useState, useRef, useEffect } from 'react';

const UPLOAD_ACCEPT = '.pdf,.txt,.text,.md,.markdown';

export default function MessageInput({ onSend, disabled, onUpload, uploading }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = '56px';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = Math.min(scrollHeight, 200) + 'px';
    }
  }, [text]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (text.trim() && !disabled && !uploading) {
        onSend(text.trim());
        setText('');
      }
    }
  };

  return (
    <div className="max-w-3xl mx-auto w-full px-4 text-center">
      <div className="relative flex items-end group bg-bgPanel border border-bgBorder rounded-2xl shadow-sm focus-within:ring-1 focus-within:ring-accentMain/50 focus-within:border-accentMain/50 transition-all duration-200">
        
        <input
          ref={fileInputRef}
          type="file"
          accept={UPLOAD_ACCEPT}
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = '';
            if (f && onUpload) onUpload(f);
          }}
        />
        <button
          type="button"
          disabled={disabled || uploading || !onUpload}
          title="Upload .pdf, .txt, or .md for RAG"
          onClick={() => fileInputRef.current?.click()}
          className="p-3 m-1 text-textMuted hover:text-accentMain transition-colors rounded-xl flex-shrink-0 disabled:opacity-40 disabled:pointer-events-none"
        >
          {uploading ? (
            <span className="inline-block w-5 h-5 border-2 border-accentMain border-t-transparent rounded-full animate-spin" />
          ) : (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13" />
            </svg>
          )}
        </button>

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            uploading
              ? 'Indexing document…'
              : disabled
                ? 'Please wait…'
                : 'Message Nova…'
          }
          disabled={disabled || uploading}
          className="flex-1 max-h-[200px] min-h-[56px] py-4 px-2 bg-transparent text-textMain placeholder-textFaint resize-none outline-none focus:ring-0 leading-relaxed custom-scrollbar disabled:opacity-50"
          rows={1}
        />
        
        <button
          onClick={() => {
            if (text.trim() && !disabled && !uploading) {
              onSend(text.trim());
              setText('');
            }
          }}
          disabled={!text.trim() || disabled || uploading}
          className={`p-2 m-2 rounded-xl flex-shrink-0 transition-all duration-200 shadow-sm
            ${text.trim() && !disabled 
              ? 'bg-textMain text-bgMain hover:opacity-90 active:scale-95' 
              : 'bg-bgHover text-textFaint'}`}
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 19V5m0 0l-7 7m7-7l7 7" />
          </svg>
        </button>
      </div>

      <p className="text-[11px] text-textFaint mt-3 opacity-60">
        Upload PDF, TXT, or Markdown to index them for <span className="text-textMuted">document_search</span>. Nova can make mistakes — verify important information.
      </p>
    </div>
  );
}
