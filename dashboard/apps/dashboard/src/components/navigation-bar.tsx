
import React from 'react';
import type { Domain } from '../types';
import { motion } from 'framer-motion';

interface NavigationBarProps {
    activeDomain: Domain;
    onActiveDomainChange: (domain: Domain) => void;
}

const DOMAINS: { id: Domain; name: string }[] = [
    { id: 'Varkiel', name: 'Varkiel Agent' },
    { id: 'Lattice', name: 'Constraint Lattice' },
    { id: 'WildCore', name: 'WildCore' },
];

export const NavigationBar: React.FC<NavigationBarProps> = ({ activeDomain, onActiveDomainChange }) => {
    return (
        <div className="border-b border-ui-primary flex flex-wrap items-center space-x-8 overflow-x-auto px-2" role="tablist">
            {DOMAINS.map(domain => {
                const isSelected = activeDomain === domain.id;
                return (
                    <button
                        key={domain.id}
                        onClick={() => onActiveDomainChange(domain.id)}
                        className={`py-4 text-sm font-medium uppercase tracking-wider relative transition-colors duration-200 ease-in-out outline-none focus-visible:text-accent-cyan ${
                            isSelected
                                ? 'text-text-primary'
                                : 'text-text-secondary hover:text-text-primary'
                        }`}
                        role="tab"
                        aria-selected={isSelected}
                    >
                        {domain.name}
                        {isSelected && (
                             <motion.div 
                                className="absolute bottom-[-1px] left-0 right-0 h-0.5 bg-accent-cyan"
                                layoutId="underline"
                            />
                        )}
                    </button>
                )
            })}
        </div>
    );
};
