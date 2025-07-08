
import React from 'react';
import type { AppState, ControllerAction, RuleID } from '../types';
import { IntroHeader } from './IntroHeader';
import { OrchestrationInputPanel } from './OrchestrationInputPanel';
import { ResultsPanel } from './ResultsPanel';
import { NavigationBar } from './NavigationBar';
import { DocumentSelectionPanel } from './DocumentSelectionPanel';
import { WizardIndicator } from './WizardIndicator';
import { VarkielPanel } from './panels/VarkielPanel';
import { LatticePanel } from './panels/LatticePanel';
import { WildCorePanel } from './panels/WildCorePanel';

interface DashboardProps {
  state: AppState;
  dispatch: React.Dispatch<ControllerAction>;
}

export const Dashboard: React.FC<DashboardProps> = ({ state, dispatch }) => {
    
  if (!state.activeFramework) {
    return <DocumentSelectionPanel dispatch={dispatch} />;
  }

  const renderDomainView = () => {
    switch (state.activeDomain) {
      case 'Varkiel':
        return <VarkielPanel state={state} dispatch={dispatch} />;
      case 'Lattice':
        return <LatticePanel state={state} dispatch={dispatch} />;
      case 'WildCore':
        return <WildCorePanel state={state} dispatch={dispatch} />;
      default:
        return null;
    }
  };

  return (
    <div className="flex flex-col space-y-8 w-full">
      <IntroHeader activeFramework={state.activeFramework} dispatch={dispatch} />

      <WizardIndicator currentStep={state.wizardStep} />
      
        <OrchestrationInputPanel
            prompt={state.prompt}
            onPromptChange={(p) => dispatch({ type: 'SET_PROMPT', payload: p })}
            onExecute={() => dispatch({ type: 'EXECUTE_NARRATIVE', payload: state.prompt })}
            isLoading={state.isLoading}
            error={state.error}
            disabled={!state.activeFramework || !state.prompt || state.wizardStep < 4}
        />

        <ResultsPanel
            finalOutput={state.finalOutput}
            introspectionReport={state.introspectionReport}
            isLoading={state.isLoading}
            affectiveData={state.affectiveData}
        />
      
      <div className="space-y-6">
        <NavigationBar
          activeDomain={state.activeDomain}
          onActiveDomainChange={(d) => dispatch({ type: 'SET_DOMAIN', payload: d })}
        />
        {renderDomainView()}
      </div>
    </div>
  );
};
