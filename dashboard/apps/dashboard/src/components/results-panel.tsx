import React from 'react';
import Card from './ui/card';
import { SpinnerIcon, WandIcon } from './icons';
import type { HighlightedOutput, IntrospectionReport, AffectiveDataPoint } from '../types';
import AffectiveChart from './charts/AffectiveChart';
import { AnimatePresence, motion } from 'framer-motion';

interface ResultsPanelProps {
  isLoading: boolean;
  finalOutput: string | HighlightedOutput;
  introspectionReport: IntrospectionReport | null;
  affectiveData: AffectiveDataPoint[];
}

const FinalOutputDisplay: React.FC<{ output: string | HighlightedOutput }> = ({ output }) => {
    if (typeof output === 'string') {
        return <>{output || <span className="text-text-secondary">Final output will appear here after pipeline execution.</span>}</>;
    }
    
    const { text, highlight } = output;
    if (!highlight || !text.includes(highlight)) {
        return <>{text}</>;
    }
    const parts = text.split(new RegExp(`(${highlight})`, 'gi'));
    return (
        <span>
            {parts.map((part, index) =>
                part.toLowerCase() === highlight.toLowerCase() ? (
                    <mark key={index} className="bg-amber-500/30 text-amber-300 px-1 py-0.5 rounded">
                        {part}
                    </mark>
                ) : (
                    <span key={index}>{part}</span>
                )
            )}
        </span>
    );
};


export const ResultsPanel: React.FC<ResultsPanelProps> = ({
  isLoading,
  finalOutput,
  introspectionReport,
  affectiveData
}) => {
  return (
    <div className="grid grid-cols-1 gap-8">
      <Card className="flex flex-col">
        <h2 className="text-xl font-semibold text-text-primary mb-4">Governed Output</h2>
        <div className="w-full flex-grow p-4 bg-background border border-white/10 rounded-md overflow-y-auto text-base leading-relaxed whitespace-pre-wrap min-h-[200px]">
          {isLoading && !finalOutput ? (
            <div className="flex items-center justify-center h-full text-text-secondary">
                <SpinnerIcon className="w-6 h-6" />
                <span className="ml-3">Processing...</span>
            </div>
          ) : (
            <FinalOutputDisplay output={finalOutput} />
          )}
        </div>
      </Card>
      
        <AnimatePresence>
        {introspectionReport && !isLoading && (
            <motion.div initial={{ opacity: 0, y:10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <Card>
                  <h2 className="text-lg font-semibold text-text-primary mb-2 flex items-center gap-2"><WandIcon className="w-5 h-5 text-accent-cyan"/>Introspection Report</h2>
                  <p className="text-base text-text-secondary italic mb-4">"{introspectionReport.summary}"</p>
                  <div className="text-sm space-y-1 text-text-secondary">
                      {introspectionReport.contributors.map(c => (
                          <p key={c.engineName}><strong>{c.engineName}:</strong> {c.impact}</p>
                      ))}
                  </div>
              </Card>
            </motion.div>
        )}
        </AnimatePresence>
    </div>
  );
};
