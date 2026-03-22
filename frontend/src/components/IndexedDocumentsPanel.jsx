import React, { useState, useEffect, useCallback } from 'react';
import { fetchIndexedDocuments, removeIndexedDocument } from '../api';

export default function IndexedDocumentsPanel({ refreshTrigger, disabled }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(true);
  const [removing, setRemoving] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    setLoading(true);
    try {
      const data = await fetchIndexedDocuments();
      setDocuments(data.documents || []);
    } catch (e) {
      setError(e?.message || 'Could not load indexed documents');
      setDocuments([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshTrigger]);

  const handleRemove = async (storedFilename) => {
    if (disabled || removing) return;
    setRemoving(storedFilename);
    setError(null);
    try {
      await removeIndexedDocument(storedFilename);
      await load();
    } catch (e) {
      setError(e?.message || 'Remove failed');
    } finally {
      setRemoving(null);
    }
  };

  return (
    <div className="max-w-3xl mx-auto w-full px-4 mb-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-2 text-left text-xs font-semibold text-textMuted uppercase tracking-wider py-2 px-3 rounded-lg border border-bgBorder bg-bgPanel/60 hover:bg-bgPanel transition-colors"
      >
        <span>
          Knowledge base ({loading ? '…' : documents.length}{' '}
          {documents.length === 1 ? 'file' : 'files'})
        </span>
        <svg
          className={`w-4 h-4 shrink-0 transition-transform ${open ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="mt-1 rounded-lg border border-bgBorder bg-black/20 overflow-hidden">
          {error && (
            <div className="text-xs text-red-300/90 px-3 py-2 border-b border-bgBorder bg-red-950/30">
              {error}
            </div>
          )}
          {loading ? (
            <p className="text-xs text-textFaint px-3 py-3">Loading…</p>
          ) : documents.length === 0 ? (
            <p className="text-xs text-textFaint px-3 py-3">
              No documents indexed yet. Upload with the paperclip — available in every chat.
            </p>
          ) : (
            <ul className="max-h-40 overflow-y-auto divide-y divide-bgBorder/80">
              {documents.map((d) => (
                <li
                  key={d.stored_filename}
                  className="flex items-center justify-between gap-2 px-3 py-2 text-sm"
                >
                  <div className="min-w-0 text-left">
                    <div className="text-textMain truncate font-medium" title={d.original_filename}>
                      {d.original_filename}
                    </div>
                    <div className="text-[11px] text-textFaint">
                      {d.chunks} chunk{d.chunks !== 1 ? 's' : ''}
                    </div>
                  </div>
                  <button
                    type="button"
                    disabled={disabled || removing === d.stored_filename}
                    onClick={() => handleRemove(d.stored_filename)}
                    className="shrink-0 text-xs px-2 py-1 rounded-md border border-red-900/50 text-red-300/90 hover:bg-red-950/40 disabled:opacity-40"
                  >
                    {removing === d.stored_filename ? '…' : 'Remove'}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
