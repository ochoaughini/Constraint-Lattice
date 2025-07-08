
import React from 'react';
import { motion } from 'framer-motion';

interface ToggleProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  colorClass?: string;
}

const Toggle: React.FC<ToggleProps> = ({ label, checked, onChange, colorClass = 'bg-accent-purple' }) => {
  return (
    <label className="flex items-center justify-between cursor-pointer group w-full">
      <span className="text-sm font-medium text-text-primary group-hover:text-white transition-colors">{label}</span>
      <div className="relative">
        <div
          className={`w-12 h-6 rounded-full transition-colors duration-200 ease-in-out ${
            checked ? colorClass : 'bg-gray-600'
          }`}
        >
          <motion.div
            className="w-4 h-4 bg-white rounded-full absolute top-1 left-1"
            layout
            transition={{ type: 'spring', stiffness: 700, damping: 30 }}
            initial={false}
            animate={{ x: checked ? '1.5rem' : '0rem' }}
          />
        </div>
      </div>
    </label>
  );
};

export default Toggle;
