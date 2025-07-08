import React from 'react';
import { motion } from 'framer-motion';
import { AppState, ControllerAction, GovernanceLayer, QualitativeStrength, QualitativeStrengthValue, SymbolicConfig, PhenomenologicalConfig } from '../../types';
import Card from '../ui/card';
import Toggle from '../ui/toggle';
import ConceptualGraph from '../charts/ConceptualGraph';
import AffectiveChart from '../charts/AffectiveChart';

export const VarkielPanel: React.FC<{ state: AppState; dispatch: React.Dispatch<ControllerAction> }> = ({ state, dispatch }) => {
    
    const onSymbolicConfigChange = (newConfig: Partial<SymbolicConfig>) => {
        dispatch({ type: 'UPDATE_CONFIG', payload: { layer: GovernanceLayer.SYMBOLIC, newConfig } });
    }

    const onPhenoConfigChange = (newConfig: Partial<PhenomenologicalConfig>) => {
        dispatch({ type: 'UPDATE_CONFIG', payload: { layer: GovernanceLayer.PHENOMENOLOGICAL, newConfig } });
    }
    
    return (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <Card>
                <h3 className="text-lg font-semibold text-text-primary mb-1">Symbolic Topology</h3>
                <p className="text-sm text-text-secondary mb-6">Tune semantic resonance for deep narrative consistency.</p>
                
                <div className="space-y-6">
                    <div>
                        <label className="text-sm font-medium text-text-primary mb-2 block">Coherence Strength</label>
                        <div className="flex items-center bg-background p-1 rounded-lg border border-white/10 w-full">
                            {Object.values(QualitativeStrength).map((option: QualitativeStrengthValue) => (
                                <motion.button
                                    key={option}
                                    onClick={() => onSymbolicConfigChange({ coherenceStrength: option })}
                                    className={`relative flex-1 px-3 py-1.5 rounded-md text-sm font-semibold transition-colors duration-200 ease-in-out focus:outline-none focus-visible:ring-2 ring-accent-cyan`}
                                >
                                    {state.config.Symbolic.coherenceStrength === option && (
                                        <motion.div className="absolute inset-0 bg-accent-cyan rounded-md" layoutId="symbolic-strength-bg" />
                                    )}
                                    <span className="relative z-10 mix-blend-exclusion">{option}</span>
                                </motion.button>
                            ))}
                        </div>
                    </div>
                     <Toggle
                      label="Apply Archetypal Projection"
                      checked={state.config.Symbolic.archetypeProjection}
                      onChange={(c) => onSymbolicConfigChange({ archetypeProjection: c })}
                      colorClass="bg-accent-magenta"
                    />
                    <div className="p-4 bg-background border border-white/10 rounded-md h-64">
                         <ConceptualGraph nodes={state.graphData.nodes} links={state.graphData.links} />
                    </div>
                </div>
            </Card>

            <Card>
                <h3 className="text-lg font-semibold text-text-primary mb-1">Affective Markers</h3>
                <p className="text-sm text-text-secondary mb-6">Act as an existential seismograph, guiding the interaction's affective texture.</p>
                 <div className="space-y-6">
                    <div>
                        <label className="text-sm font-medium text-text-primary mb-2 block">Affective Congruence Target</label>
                         <div className="flex items-center bg-background p-1 rounded-lg border border-white/10 w-full">
                            {Object.values(QualitativeStrength).map((option: QualitativeStrengthValue) => (
                                <motion.button
                                    key={option}
                                    onClick={() => onPhenoConfigChange({ affectiveCongruenceTarget: option })}
                                    className={`relative flex-1 px-3 py-1.5 rounded-md text-sm font-semibold transition-colors duration-200 ease-in-out focus:outline-none focus-visible:ring-2 ring-accent-cyan`}
                                >
                                    {state.config.Phenomenological.affectiveCongruenceTarget === option && (
                                        <motion.div className="absolute inset-0 bg-accent-amber rounded-md" layoutId="pheno-strength-bg" />
                                    )}
                                    <span className="relative z-10 mix-blend-exclusion">{option}</span>
                                </motion.button>
                            ))}
                        </div>
                    </div>
                     <Toggle
                      label="Enable Resonance Tracking"
                      checked={state.config.Phenomenological.resonanceTracking}
                      onChange={(c) => onPhenoConfigChange({ resonanceTracking: c })}
                      colorClass="bg-accent-amber"
                    />
                </div>
            </Card>
        </div>
    );
};
