
import React from 'react';
import { GithubIcon } from './icons';

const Header: React.FC = () => {
  return (
    <header className="bg-background/80 backdrop-blur-sm border-b border-ui-primary sticky top-0 z-50">
      <div className="max-w-viewport mx-auto px-4 sm:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <span className="text-lg font-semibold tracking-tight text-text-primary">
              Constraint-Lattice
            </span>
          </div>
          <a
            href="https://github.com/ochoaughini/Constraint-Lattice"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center space-x-2 text-text-secondary hover:text-text-primary transition-colors duration-200"
            aria-label="View on GitHub"
          >
            <GithubIcon className="w-5 h-5" />
            <span className="hidden sm:block text-sm font-medium">View on GitHub</span>
          </a>
        </div>
      </div>
    </header>
  );
};

export default Header;
