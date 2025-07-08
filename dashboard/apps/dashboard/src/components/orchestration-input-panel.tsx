
import React from 'react';
import { PlayIcon, SpinnerIcon } from './icons';
import { motion } from 'framer-motion';

interface OrchestrationInputPanelProps {
    prompt: string;
    onPromptChange: (prompt: string) => void;
    onExecute: () => void;
    isLoading: boolean;
    error: string | null;
    disabled: boolean;
}

export const OrchestrationInputPanel: React.FC<OrchestrationInputPanelProps> = ({ prompt, onPromptChange, onExecute, isLoading, error, disabled }) => {
    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onExecute();
    };

    return (
        <form onSubmit={handleSubmit} className="space-y-4">
            <textarea
                value={prompt}
                onChange={(e) => onPromptChange(e.target.value)}
                placeholder="Enter your existential prompt here..."
                className="w-full p-4 bg-ui-primary border border-white/10 rounded-lg focus:ring-2 focus:ring-accent-cyan focus:outline-none transition-all duration-150 resize-none font-mono text-base h-24 shadow-inner"
                rows={3}
                disabled={isLoading}
                aria-label="Prompt Input"
            />
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-end gap-2">
                 {error && <p className="text-red-400 text-sm text-center flex-1">{error}</p>}
                <motion.button
                    type="submit"
                    disabled={isLoading || disabled}
                    className="px-6 py-3 bg-gradient-to-r from-accent-cyan to-accent-blue text-background font-semibold text-base rounded-md transition-all duration-150 ease-in-out flex items-center justify-center gap-3
                    disabled:bg-none disabled:bg-white/10 disabled:text-text-secondary disabled:cursor-not-allowed"
                    whileHover={{ scale: disabled ? 1 : 1.02, boxShadow: disabled ? 'none' : '0px 0px 8px rgba(0,224,224,0.5)' }}
                    whileTap={{ scale: disabled ? 1 : 0.98 }}
                >
                    {isLoading ? <SpinnerIcon /> : <PlayIcon className="w-5 h-5" />}
                    {isLoading ? 'Executing...' : 'Execute Narrative'}
                </motion.button>
            </div>
        </form>
    );
};
