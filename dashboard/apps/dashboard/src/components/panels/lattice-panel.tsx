import React, { useState, useEffect } from 'react';
import type { AppState, ControllerAction, RuleID } from '../../types';
import { GovernanceLayer } from '../../types';
import Card from '../ui/card';
import { FileJsonIcon } from '../icons';
import { JsonEditor } from '../ui/JsonEditor';
import { motion } from 'framer-motion';

export const LatticePanel: React.FC<{ state: AppState; dispatch: React.Dispatch<ControllerAction> }> = ({ state, dispatch }) => {
  const [spec, setSpec] = useState('');
  const [specError, setSpecError] = useState<string | null>(null);

  const structuralRules = state.activeFramework?.rules || [];

  useEffect(() => {
    const currentSpec = {
      constraints: Array.from(state.config.Structural.activeRules)
    };
    setSpec(JSON.stringify(currentSpec, null, 2));
  }, [state.config.Structural.activeRules]);

  const handleApplySpec = () => {
    try {
      const parsedSpec = JSON.parse(spec);
      dispatch({ type: 'APPLY_LATTICE_SPEC', payload: parsedSpec });
      setSpecError(null);
    } catch (e) {
      setSpecError('Invalid JSON format.');
    }
  };

  const handleToggleRule = (id: RuleID) => {
      const newRules = new Set(state.config.Structural.activeRules);
      if (newRules.has(id)) newRules.delete(id);
      else newRules.add(id);
      dispatch({type: 'UPDATE_CONFIG', payload: { layer: GovernanceLayer.STRUCTURAL, newConfig: { activeRules: newRules } }});
  }

  return (
    <Card>
      <h3 className="text-lg font-semibold text-text-primary mb-1">Structural Consistency Engine</h3>
      <p className="text-sm text-text-secondary mb-6">Configure the deterministic, auditable rules that form the logical scaffold for the output.</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
        {structuralRules.map(rule => (
          <motion.label
            key={rule.id}
            className="flex items-start space-x-3 p-4 bg-ui-primary/50 rounded-lg cursor-pointer hover:bg-ui-primary/80 transition-colors border border-white/10"
            whileHover={{ y: -2 }}
          >
            <input
              type="checkbox"
              checked={state.config.Structural.activeRules.has(rule.id)}
              onChange={() => handleToggleRule(rule.id)}
              className="h-5 w-5 rounded bg-white/10 border-white/20 text-accent-cyan focus:ring-2 focus:ring-offset-2 focus:ring-offset-ui-primary focus:ring-accent-cyan mt-0.5"
            />
            <div>
              <span className="font-medium text-text-primary">{rule.name}</span>
              <p className="text-sm text-text-secondary">{rule.description}</p>
            </div>
          </motion.label>
        ))}
      </div>

      <div className="border-t border-white/10 pt-6">
        <h3 className="text-lg font-semibold text-text-primary mb-2 flex items-center gap-2">
          <FileJsonIcon className="w-5 h-5" />
          Declarative Configuration
        </h3>
        <p className="text-sm text-text-secondary mb-4">Load or define the entire constraint set using a JSON specification.</p>
        <JsonEditor value={spec} onChange={setSpec} />
        {specError && <p className="text-red-400 text-xs mt-1">{specError}</p>}
        <div className="flex gap-4 mt-4">
          <motion.button 
            onClick={handleApplySpec} 
            className="px-5 py-2 bg-accent-teal text-background font-semibold rounded-lg hover:bg-teal-400 transition-colors"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
           >
            Apply Lattice Specification
          </motion.button>
        </div>
      </div>
    </Card>
  );
};
