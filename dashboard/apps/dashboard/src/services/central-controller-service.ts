import { v4 as uuidv4 } from 'uuid';
import {
  AppState,
  ControllerAction,
  ControllerEvent,
  ENGINE_NAMES,
  EngineName,
  GovernanceLayerType,
  GovernanceLayer,
  DocumentFramework,
  PipelineLogEntry,
  IntrospectionReport,
  RuleID
} from '../types';

// Define core types locally to avoid import issues
export const QualitativeStrength = {
  WEAK: 'weak',
  MEDIUM: 'medium',
  STRONG: 'strong',
} as const;

export type QualitativeStrengthValue = (typeof QualitativeStrength)[keyof typeof QualitativeStrength];

// Define governance layer constants
// const GovernanceLayer = {
//   STRUCTURAL: 'Structural',
//   SYMBOLIC: 'Symbolic',
//   PHENOMENOLOGICAL: 'Phenomenological',
//   SECURITY: 'Security',
// } as const;

// type GovernanceLayerType = typeof GovernanceLayer[keyof typeof GovernanceLayer];

// Define engine names
// const ENGINE_NAMES = {
//   [GovernanceLayer.STRUCTURAL]: 'Constraint Lattice',
//   [GovernanceLayer.SYMBOLIC]: 'Varkiel',
//   [GovernanceLayer.PHENOMENOLOGICAL]: 'Varkiel',
//   [GovernanceLayer.SECURITY]: 'WildCore',
//   'LLM': 'LLM',
//   'System': 'System',
//   'User': 'User',
//   'Constitution': 'Constitution',
//   'Bitcoin': 'Bitcoin',
//   'ACLS': 'ACLS',
// } as const;

// type EngineName = keyof typeof ENGINE_NAMES;

// Define document framework interface
// interface DocumentFramework {
//   id: DocumentFrameworkId;
//   title: string;
//   source: string;
//   description: string;
//   thumbnail: string;
//   filePath: string;
//   totalPages: number;
//   toc: Array<{ title: string; page: number }>;
//   samplePrompts: Array<{ id: string; text: string }>;
//   rules: readonly GovernanceRule[];
// }

// Define governance rule interface
// interface GovernanceRule {
//   readonly id: RuleID;
//   readonly name: string;
//   readonly description: string;
//   readonly layer: GovernanceLayerType;
// }

// Define configuration interfaces
interface StructuralConfig {
  activeRules: Set<RuleID>;
}

interface SymbolicConfig {
  coherenceStrength: QualitativeStrengthValue;
  archetypeProjection: boolean;
}

interface PhenomenologicalConfig {
  affectiveCongruenceTarget: QualitativeStrengthValue;
  resonanceTracking: boolean;
}

interface SecurityConfig {
  injectionSensitivity: QualitativeStrengthValue;
  simulationEnabled: boolean;
}

interface GovernanceConfig {
  [GovernanceLayer.STRUCTURAL]: StructuralConfig;
  [GovernanceLayer.SYMBOLIC]: SymbolicConfig;
  [GovernanceLayer.PHENOMENOLOGICAL]: PhenomenologicalConfig;
  [GovernanceLayer.SECURITY]: SecurityConfig;
}

// Define pipeline log entry
// interface PipelineLogEntry {
//   id: string;
//   timestamp: Date;
//   timeOffset: number;
//   module: EngineName;
//   event: string;
//   details: string;
//   metadata?: Record<string, any>;
// }

// Define introspection report
// interface IntrospectionReport {
//   id: string;
//   framework: DocumentFrameworkId;
//   timestamp: string;
//   strengths: string[];
//   weaknesses: string[];
//   contributors: Array<{ id: string; name: string; contribution: string }>;
//   summary: string;
// }

// Define app state
// interface AppState {
//   activeFramework: DocumentFramework | null;
//   wizardStep: number;
//   activeDomain: Domain;
//   prompt: string;
//   finalOutput: string | { text: string; highlight: string; type?: string };
//   config: GovernanceConfig;
//   logs: PipelineLogEntry[];
//   graphData: { nodes: any[]; links: any[] };
//   nodes: any[];
//   links: any[];
//   affectiveData: any[];
//   introspectionReport: IntrospectionReport | null;
//   isLoading: boolean;
//   error: string | null;
// }

// Define controller action types
// type ControllerAction = 
//   | { type: 'SET_FRAMEWORK'; payload: DocumentFramework }
//   | { type: 'ADD_LOG'; payload: PipelineLogEntry }
//   | { type: 'SET_REPORT'; payload: IntrospectionReport }
//   | { type: 'LOAD_DOCUMENT'; payload: DocumentFrameworkId }
//   | { type: 'RESET_FRAMEWORK' }
//   | { type: 'SET_PROMPT'; payload: string }
//   | { type: 'SET_DOMAIN'; payload: Domain }
//   | { type: 'UPDATE_CONFIG'; payload: { layer: GovernanceLayerType; newConfig: any } }
//   | { type: 'APPLY_LATTICE_SPEC'; payload: any }
//   | { type: 'EXECUTE_NARRATIVE'; payload: string }
//   | { type: 'EXECUTE_BREACH_SIM'; payload: string };

// Helper function to create a fresh initial state
const getFreshInitialState = (): AppState => ({
  prompt: '',
  activeFramework: undefined,
  activeDomain: '',
  wizardStep: 0,
  config: {
    [GovernanceLayer.STRUCTURAL]: { activeRules: new Set() },
    [GovernanceLayer.SYMBOLIC]: { 
      coherenceStrength: QualitativeStrength.WEAK, 
      archetypeProjection: false 
    },
    [GovernanceLayer.PHENOMENOLOGICAL]: { 
      affectiveCongruenceTarget: QualitativeStrength.WEAK, 
      resonanceTracking: false 
    },
    [GovernanceLayer.SECURITY]: { 
      injectionSensitivity: QualitativeStrength.WEAK, 
      simulationEnabled: false 
    },
  },
  introspectionReport: undefined,
  logs: [],
  graphData: { nodes: [], links: [] },
  nodes: [],
  links: [],
  affectiveData: [],
  isLoading: false,
  error: null,
  finalOutput: '',
});

type StateListener = (state: AppState) => void;

export class CentralControllerService {
  private state: AppState;
  private listeners: Set<StateListener> = new Set();
  private runStartTime: number | null = null;

  constructor() {
    this.state = getFreshInitialState();
  }

  // Get the current state
  public getState(): AppState {
    return { ...this.state };
  }

  // Get the initial state
  public getInitialState(): AppState {
    return getFreshInitialState();
  }

  // Set state with partial update
  public setState(updater: Partial<AppState> | ((prevState: AppState) => Partial<AppState>)): void {
    const nextPartialState = typeof updater === 'function' ? updater(this.state) : updater;
    this.state = { ...this.state, ...nextPartialState };
    this.notifyStateChange();
  }

  // Subscribe to state updates
  public on(event: ControllerEvent, listener: StateListener): void {
    this.listeners.add(listener);
    // Immediately call with current state
    listener(this.state);
  }

  // Unsubscribe from state updates
  public off(event: ControllerEvent, listener: StateListener): void {
    this.listeners.delete(listener);
  }

  // Notify all listeners of state changes
  private notifyStateChange(): void {
    this.listeners.forEach(listener => listener(this.state));
  }

  // Handle action
  public dispatch(action: ControllerAction): void {
    this.setState(prevState => this.reduce(prevState, action));
  }

  // Reducer function to handle state updates based on actions
  private reduce(state: AppState, action: ControllerAction): AppState {
    switch (action.type) {
      case 'SET_FRAMEWORK':
        return {
          ...state,
          activeFramework: action.payload,
        };
      case 'ADD_LOG':
        return {
          ...state,
          logs: [...state.logs, action.payload],
        };
      case 'SET_REPORT':
        return {
          ...state,
          introspectionReport: action.payload,
        };
      case 'LOAD_DOCUMENT':
        // Handle document loading logic here
        return state;
      case 'RESET_FRAMEWORK':
        return getFreshInitialState();
      case 'SET_PROMPT':
        return { ...state, prompt: action.payload };
      case 'SET_DOMAIN':
        return { ...state, activeDomain: action.payload };
      case 'UPDATE_CONFIG':
        return {
          ...state,
          config: {
            ...state.config,
            [action.payload.layer]: action.payload.newConfig,
          },
        };
      case 'EXECUTE_BREACH_SIM':
        return {
          ...state,
          isLoading: true,
        };
      default:
        return state;
    }
  }

  // Add a log entry
  public addLog(entry: Omit<PipelineLogEntry, 'id' | 'timestamp' | 'timeOffset'>): void {
    const newEntry: PipelineLogEntry = {
      id: uuidv4(),
      timestamp: new Date(),
      timeOffset: this.runStartTime ? Date.now() - this.runStartTime : 0,
      ...entry
    };

    this.setState(prevState => ({
      logs: [newEntry, ...(prevState.logs || [])].slice(0, 1000)
    }));
  }

  // Set the introspection report
  public setIntrospectionReport(report: IntrospectionReport): void {
    this.setState({ introspectionReport: report });
  }

  // Set loading state
  public setLoading(isLoading: boolean): void {
    this.setState({ isLoading });
  }

  // Set error
  public setError(error: string | null): void {
    this.setState({ error });
  }

  // Reset the service to initial state
  public reset(): void {
    this.state = getFreshInitialState();
    this.notifyStateChange();
  }
}

// Export a singleton instance
export const centralControllerService = new CentralControllerService();
