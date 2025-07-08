
import React from 'react';
import { DOCUMENT_FRAMEWORKS } from '../governance';
import type { ControllerAction } from '../types';
import { FrameworkCard } from './FrameworkCard';
import { IntroHeader } from './IntroHeader';
import Footer from './Footer';

interface DocumentSelectionPanelProps {
  dispatch: React.Dispatch<ControllerAction>;
}

export const DocumentSelectionPanel: React.FC<DocumentSelectionPanelProps> = ({ dispatch }) => {
  return (
    <div className="flex flex-col min-h-screen px-4 space-y-8">
      <IntroHeader activeFramework={null} dispatch={dispatch} />
      <div className="flex-grow text-center py-16">
        <h2 className="text-2xl font-semibold text-text-primary tracking-tight">Select a Constraint Framework</h2>
        <p className="mt-4 text-lg text-text-secondary max-w-2xl mx-auto">
          Choose a foundational document to act as the active constraint set for the governance pipeline.
        </p>
        <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-4xl mx-auto">
          {Object.values(DOCUMENT_FRAMEWORKS).map(framework => (
            <FrameworkCard 
              key={framework.id}
              framework={framework}
              onSelect={() => dispatch({ type: 'LOAD_DOCUMENT', payload: framework.id })}
            />
          ))}
        </div>
        <div className="mt-12 text-center">
              <h3 className="text-lg font-semibold text-text-primary">Or Upload Your Own</h3>
              <div className="mt-4 max-w-lg mx-auto p-8 border-2 border-dashed border-gray-600 rounded-lg text-gray-500 hover:border-accent-cyan hover:text-accent-cyan transition-colors cursor-pointer">
                  PDF drop zone placeholder.
              </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};
