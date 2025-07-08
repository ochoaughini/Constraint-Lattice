
import React, { createContext, useContext, useMemo } from 'react';
import { CentralControllerService } from '../services/central-controller-service';

export const ControllerContext = createContext<CentralControllerService | null>(null);

export const ControllerProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const controllerService = useMemo(() => new CentralControllerService(), []);

    return (
        <ControllerContext.Provider value={controllerService}>
            {children}
        </ControllerContext.Provider>
    );
};
