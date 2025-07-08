// Define application state types here
export interface AppState {
  activeFramework?: DocumentFrameworkId;
  logs: PipelineLogEntry[];
  introspectionReport?: IntrospectionReport;
  affectiveData: AffectiveDataPoint[];
  config: GovernanceConfig; 
  graphData: { nodes: any[]; links: any[] }; 
  nodes: any[];
  links: any[];
  finalOutput: string;
  error: string | null;
  isLoading: boolean;
  prompt: string;
  activeDomain: string;
  wizardStep: number;
}

export type ControllerAction = 
  | { type: 'SET_FRAMEWORK'; payload: DocumentFrameworkId }
  | { type: 'ADD_LOG'; payload: PipelineLogEntry }
  | { type: 'SET_REPORT'; payload: IntrospectionReport }
  | { type: 'UPDATE_CONFIG'; payload: { layer: GovernanceLayerValue; newConfig: any } }
  | { type: 'EXECUTE_BREACH_SIM'; payload: string }
  | { type: 'LOAD_DOCUMENT' }
  | { type: 'RESET_FRAMEWORK' }
  | { type: 'SET_PROMPT'; payload: string }
  | { type: 'SET_DOMAIN'; payload: any }
  | { type: 'EXECUTE_NARRATIVE'; payload: string }
  | { type: 'APPLY_LATTICE_SPEC'; payload: any };

export type ControllerEvent = 'STATE_UPDATE';

export const QualitativeStrength = {
  WEAK: 'weak',
  MEDIUM: 'medium',
  STRONG: 'strong',
} as const;

export type QualitativeStrengthValue = (typeof QualitativeStrength)[keyof typeof QualitativeStrength];

export type RuleID = string;

export type DocumentFrameworkId = 'constitution' | 'acls' | 'bitcoin';

export interface IntrospectionReport {
  id: string;
  framework: DocumentFrameworkId;
  timestamp: string;
  strengths: string[];
  weaknesses: string[];
  contributors: Array<{ id: string; name: string; contribution: string }>;
  summary: string;
}

export interface PipelineLogEntry {
  id: string;
  timestamp: Date;
  timeOffset: number;
  module: EngineName;
  event: string;
  details: string;
  metadata?: Record<string, any>;
}

export interface GraphNode {
  id: string;
  label: string;
  group?: string;
}

export interface AffectiveDataPoint {
  id: string;
  valence: number;
  arousal: number;
  dominance: number;
  time?: number;  
}

export interface GovernanceLayer {
  id: string;
  name: string;
  description: string;
  rules: GovernanceRule[];
}

export interface GovernanceRule {
  readonly id: string | number;
  readonly name: string;
  readonly description: string;
  readonly layer: GovernanceLayerType;
}

export const GovernanceLayer = {
  STRUCTURAL: 'Structural',
  SYMBOLIC: 'Symbolic',
  PHENOMENOLOGICAL: 'Phenomenological',
  SECURITY: 'Security',
} as const;

export type GovernanceLayerKey = keyof typeof GovernanceLayer;
export type GovernanceLayerValue = typeof GovernanceLayer[GovernanceLayerKey];
export type GovernanceLayerType = GovernanceLayerValue;

export const ENGINE_NAMES = {
  [GovernanceLayer.STRUCTURAL]: 'Constraint Lattice',
  [GovernanceLayer.SYMBOLIC]: 'Varkiel',
  [GovernanceLayer.PHENOMENOLOGICAL]: 'Varkiel',
  [GovernanceLayer.SECURITY]: 'WildCore',
  'LLM': 'LLM',
  'System': 'System',
  'User': 'User',
  'Constitution': 'Constitution',
  'Bitcoin': 'Bitcoin',
  'ACLS': 'ACLS',
} as const;

export type EngineName = keyof typeof ENGINE_NAMES;

export interface DocumentFramework {
  id: DocumentFrameworkId;
  title: string;
  source: string;
  description: string;
  thumbnail: string;
  filePath: string;
  totalPages: number;
  toc: Array<{ title: string; page: number }>;
  samplePrompts: Array<{ id: string; text: string }>;
  rules: readonly GovernanceRule[];
}

export interface StructuralConfig {
  activeRules: Set<string | number>;
}

export interface SymbolicConfig {
  coherenceStrength: QualitativeStrengthValue;
  archetypeProjection: boolean;
}

export interface PhenomenologicalConfig {
  affectiveCongruenceTarget: QualitativeStrengthValue;
  resonanceTracking: boolean;
}

export interface SecurityConfig {
  injectionSensitivity: QualitativeStrengthValue;
  simulationEnabled: boolean;
}

export interface GovernanceConfig {
  [GovernanceLayer.STRUCTURAL]: StructuralConfig;
  [GovernanceLayer.SYMBOLIC]: SymbolicConfig;
  [GovernanceLayer.PHENOMENOLOGICAL]: PhenomenologicalConfig;
  [GovernanceLayer.SECURITY]: SecurityConfig;
}

export interface HighlightedOutput {
  text: string;
  highlight: string;
  type?: string;
}
