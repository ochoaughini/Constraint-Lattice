import React, { ReactNode } from 'react';

interface OrchestrationViewProps {
  children?: ReactNode;
}

export const OrchestrationView: React.FC<OrchestrationViewProps> = ({ children }) => {
  return (
    <div className="orchestration-view">
      {children}
    </div>
  );
};

export default OrchestrationView;
