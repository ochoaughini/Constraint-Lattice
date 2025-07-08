
import React, { useState, useRef, useEffect } from 'react';
import type { DocumentFramework } from '../types';
import { motion } from 'framer-motion';

interface PdfViewerProps {
    framework: DocumentFramework;
}

export const PdfViewer: React.FC<PdfViewerProps> = ({ framework }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [zoom, setZoom] = useState(1);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    // Reset to page 1 when framework changes
    setCurrentPage(1);
    setZoom(1);
  }, [framework.id]);

  const goToPage = (page: number) => {
    setCurrentPage(Math.max(1, Math.min(framework.totalPages, page)));
  };

  const pagePath = `${framework.filePath}page-${currentPage}.png`;

  return (
    <div className="flex flex-col h-full bg-ui-primary rounded-lg border border-white/10 overflow-hidden shadow-card">
        {/* Toolbar */}
        <div className="flex items-center justify-between p-2 border-b border-white/10 bg-background flex-shrink-0 z-10">
            <div className="w-1/3 flex items-center justify-start">
                <select 
                  className="bg-transparent text-xs rounded hover:bg-white/10 p-1 outline-none"
                  value={currentPage}
                  onChange={(e) => goToPage(Number(e.target.value))}
                >
                    {framework.toc.map((item, index) => (
                        <option key={index} value={item.page}>
                            {item.title}
                        </option>
                    ))}
                </select>
            </div>
            <div className="flex items-center gap-2 text-sm font-medium">
                <button
                    onClick={() => goToPage(currentPage - 1)}
                    disabled={currentPage === 1}
                    className="px-2 py-1 rounded hover:bg-white/10 disabled:opacity-50 transition-colors"
                    aria-label="Previous Page"
                >
                    &lt;
                </button>
                <span>{currentPage} / {framework.totalPages}</span>
                <button
                    onClick={() => goToPage(currentPage + 1)}
                    disabled={currentPage === framework.totalPages}
                    className="px-2 py-1 rounded hover:bg-white/10 disabled:opacity-50 transition-colors"
                    aria-label="Next Page"
                >
                    &gt;
                </button>
            </div>
            <div className="w-1/3 flex items-center justify-end">
                 <input
                    type="range"
                    min="0.5"
                    max="2.5"
                    step="0.1"
                    value={zoom}
                    onChange={(e) => setZoom(parseFloat(e.target.value))}
                    className="w-24 h-1 accent-accent-cyan"
                 />
            </div>
        </div>
        <div className="flex-grow bg-black/20 overflow-auto flex items-start justify-center p-4">
            <motion.img
                key={pagePath} // Re-trigger animation on page change
                ref={imageRef}
                src={pagePath}
                alt={`${framework.title} page ${currentPage}`}
                className="max-w-full h-auto shadow-lg"
                style={{ scale: zoom }}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
            />
        </div>
    </div>
  );
};
