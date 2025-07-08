import React, { createContext, useContext, ReactNode } from 'react';
import { useCentralController } from '../hooks/useCentralController';

interface ControllerContextType {
  state: {
    activeFramework: any; // Replace 'any' with the appropriate type for your framework
  };
  updateFramework: (framework: any) => void;
}

const ControllerContext = createContext<ControllerContextType | undefined>(undefined);

const ControllerProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { state, updateFramework } = useCentralController();

  return (
    <ControllerContext.Provider value={{ state, updateFramework }}>
      {children}
    </ControllerContext.Provider>
  );
};

const useController = (): ControllerContextType => {
  const context = useContext(ControllerContext);
  if (context === undefined) {
    throw new Error('useController must be used within a ControllerProvider');
  }
  return context;
};

export { useController };
export default ControllerProvider;
