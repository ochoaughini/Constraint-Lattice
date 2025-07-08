import { useState, useCallback } from 'react';

interface CentralControllerState {
  activeFramework: any; // Replace 'any' with the appropriate type for your framework
  // Add other state properties as needed
}

export const useCentralController = () => {
  const [state, setState] = useState<CentralControllerState>({
    activeFramework: null,
    // Initialize other state properties here
  });

  // Add any controller methods here
  const updateFramework = useCallback((framework: any) => {
    setState(prev => ({
      ...prev,
      activeFramework: framework
    }));
  }, []);

  return {
    state,
    updateFramework,
    // Export other methods as needed
  };
};

export default useCentralController;
