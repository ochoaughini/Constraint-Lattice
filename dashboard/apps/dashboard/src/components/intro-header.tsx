
import React from 'react';
import type { DocumentFramework, ControllerAction } from '../types';

interface IntroHeaderProps {
    activeFramework: DocumentFramework | null;
    dispatch: React.Dispatch<ControllerAction>;
}

export const IntroHeader: React.FC<IntroHeaderProps> = ({ activeFramework, dispatch }) => {
    if (!activeFramework) {
        return (
            <section className="py-12 border-b border-ui-primary">
                <div className="max-w-3xl mx-auto text-center">
                    <h1 className="text-3xl font-semibold text-text-primary tracking-tight">Constitutional AI, Operationalized</h1>
                    <p className="mt-4 text-lg text-text-secondary">
                        A live, auditable AI interface governed by real-world normative structures.
                    </p>
                </div>
            </section>
        );
    }
    
    return (
        <section className="text-center">
            <h1 className="text-2xl font-semibold text-text-primary tracking-tight">Constraint-Lattice Governance</h1>
            <p className="mt-2 text-md text-text-secondary">
                Active Framework: <span className="text-accent-cyan font-semibold">{activeFramework.title}</span>
                <button 
                  onClick={() => dispatch({ type: 'RESET_FRAMEWORK'})} 
                  className="ml-2 text-xs text-accent-blue hover:underline"
                >
                    (Change)
                </button>
            </p>
        </section>
    );
};
