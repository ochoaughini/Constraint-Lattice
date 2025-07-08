import { useState, useEffect, useContext } from 'react';
import type { AppState, ControllerAction, ControllerEvent } from '../types';
import { ControllerContext } from '../contexts/controller-context';

export const useCentralController = () => {
  const service = useContext(ControllerContext) as any; // Temporary any to bypass type checking
  if (!service) {
    throw new Error('useCentralController must be used within a ControllerProvider');
  }

  const [state, setState] = useState<AppState>(service.getInitialState());

  useEffect(() => {
    const handleStateUpdate = (newState: Partial<AppState>) => {
      setState(prev => ({ ...prev, ...newState }));
    };

    service.on('STATE_UPDATE', handleStateUpdate);
    
    return () => {
      service.off('STATE_UPDATE', handleStateUpdate);
    };
  }, [service]);
  
  const dispatch = (action: ControllerAction) => {
    service.dispatch(action);
  };

  return { state, dispatch };
};
