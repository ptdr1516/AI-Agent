import React, { useState, useRef, useEffect } from 'react';

const UPLOAD_ACCEPT = '.pdf,.txt,.text,.md,.markdown';

export default function MessageInput({ onSend, disabled, onUpload, uploading, editMessage, onCancelEdit }) {
  const [text, setText] = useState('');
  const textareaRef = useRef(null);
  const fileInputRef = useRef(null);

  // When editMessage changes, set the text to edit
  useEffect(() => {
    if (editMessage) {
      setText(editMessage.content);
      // Focus the textarea
      setTimeout(() => {
        textareaRef.current?.focus();
        textareaRef.current?.setSelectionRange(textareaRef.current.value.length, textareaRef.current.value.length);
      }, 0);
    } else {
      // Clear text when not editing (conversation switch)
      setText('');
    }
  }, [editMessage]);

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
    } else if (e.key === 'Escape' && editMessage) {
      // Cancel edit on Escape
      onCancelEdit && onCancelEdit();
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-2 sm:px-4">
      <div className="relative flex items-end gap-1.5 sm:gap-3 bg-bgHover border border-bgBorder p-1.5 sm:p-2 rounded-2xl shadow-lg focus-within:border-accentMain/50 focus-within:ring-1 focus-within:ring-accentMain/20 transition-all duration-200">
        
        {/* Edit mode indicator */}
        {editMessage && (
          <div className="absolute -top-10 left-0 right-0 flex items-center justify-between bg-accentMain/10 border border-accentMain/30 rounded-lg px-3 py-2 animate-fade-in">
            <div className="flex items-center gap-2 text-xs text-accentMain">
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
              </svg>
              <span className="font-semibold">Editing message</span>
            </div>
            <button
              onClick={onCancelEdit}
              className="text-xs text-textMuted hover:text-textMain underline"
            >
              Cancel
            </button>
          </div>
        )}
        
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
          placeholder={uploading ? "Uploading..." : "Message Nova..."}
          disabled={disabled || uploading}
          className="w-full max-h-[200px] min-h-[56px] bg-transparent text-textMain text-base placeholder-textFaint px-4 py-4 resize-none focus:outline-none custom-scrollbar disabled:opacity-50"
          style={{ overflowY: 'auto' }}
        />
        
        <button
          onClick={() => {
            if (text.trim() && !disabled && !uploading) {
              onSend(text.trim());
              setText('');
            }
          }}
          disabled={!text.trim() || disabled || uploading}
          className={`p-3 m-1 rounded-xl flex-shrink-0 transition-all duration-200 shadow-sm
            ${text.trim() && !disabled 
              ? 'bg-textMain text-bgMain hover:opacity-90 active:scale-95' 
              : 'bg-bgHover text-textFaint'}`}
        >
          {editMessage ? (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 19V5m0 0l-7 7m7-7l7 7" />
            </svg>
          )}
        </button>
      </div>

      <p className="text-[11px] text-textFaint mt-3 opacity-60">
        Upload PDF, TXT, or Markdown to index them for <span className="text-textMuted">document_search</span>. Nova can make mistakes — verify important information.
      </p>
    </div>
  );
}
