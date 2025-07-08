
import React from 'react';
import { motion } from 'framer-motion';

interface WizardIndicatorProps {
  currentStep: number;
}

const steps = [
  "Select Document",
  "Configure Constraints",
  "Tune Varkiel Agent",
  "Execute & Audit"
];

export const WizardIndicator: React.FC<WizardIndicatorProps> = ({ currentStep }) => {
  return (
    <div className="flex flex-wrap items-center justify-between w-full gap-2 py-4">
      {steps.map((step, index) => {
        const stepNumber = index + 1;
        const isActive = currentStep === stepNumber;
        const isCompleted = currentStep > stepNumber;

        return (
          <React.Fragment key={step}>
            <div className="flex flex-col items-center text-center px-2">
              <motion.div
                className="w-8 h-8 rounded-full border-2 flex items-center justify-center font-semibold transition-colors duration-300 flex-shrink-0"
                animate={isCompleted ? "completed" : isActive ? "active" : "inactive"}
                variants={{
                  completed: { backgroundColor: '#38B2AC', borderColor: '#38B2AC', color: '#0E1117' },
                  active: { backgroundColor: 'transparent', borderColor: '#38B2AC', color: '#38B2AC' },
                  inactive: { backgroundColor: 'transparent', borderColor: '#4A5568', color: '#A0AEC0' }
                }}
              >
                {isCompleted ? '✓' : stepNumber}
              </motion.div>
              <p className={`mt-2 text-xs font-medium transition-colors duration-300 ${isCompleted || isActive ? 'text-text-primary' : 'text-text-tertiary'}`}>
                {step}
              </p>
            </div>
            {index < steps.length - 1 && (
              <div className="flex-1 h-0.5 bg-gray-700 " />
            )}
          </React.Fragment>
        );
      })}
    </div>
  );
};
