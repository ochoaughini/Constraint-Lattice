"""Phenomenological Tracker - Affect-state resonance modeling

This module models cognitive-affective dynamics by encoding memory
engrams as PatternResonanceVectors within a high-dimensional resonance
manifold. It maintains dimensional consistency across temporal slices.

Key Components:
- PatternResonanceVectors: Tracks memory states as resonance vectors
- State Projection: Maintains dimensional consistency

Performance Notes:
- Incremental SVD for dimensionality reduction
- Streaming updates for memory efficiency
"""
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import IncrementalPCA
import uuid
import logging
from typing import Optional
from state_vector import StateVector  # Ensure this import exists
from collections import deque

class PatternResonanceVectors:
    """Represents memory engrams as dynamically evolving resonance vectors.
    
    Attributes:
        vectors: Matrix of resonance vectors (n_vectors x dimension)
        decay_factor: Exponential decay factor for old memories
    """
    def __init__(self, dimension: int = 256, decay_factor: float = 0.01, lattice_wrapper=None):
        self.dimension = dimension
        self.decay_factor = decay_factor
        self.vectors = np.zeros((0, dimension))
        self.lattice_wrapper = lattice_wrapper
        self.symbolic_region_history = []
        
    def update(self, new_vector: np.ndarray):
        """Update resonance vectors with a new observation.
        
        Args:
            new_vector: New state vector to incorporate
        """
        if self.vectors.size == 0:
            self.vectors = new_vector[np.newaxis, :]
        else:
            # Apply decay to existing vectors
            self.vectors *= (1 - self.decay_factor)
            # Add new vector
            self.vectors = np.vstack([self.vectors, new_vector])
        if self.lattice_wrapper:
            symbolic_coordinates = self.lattice_wrapper.get_symbolic_coordinates(new_vector)
            self.symbolic_region_history.append(symbolic_coordinates)
            
    def project(self, target_dimension: int = 3) -> np.ndarray:
        """Project high-dimensional vectors to lower dimension for visualization.
        
        Args:
            target_dimension: Output dimension (typically 2 or 3)
            
        Returns:
            Projected vectors
        """
        if self.vectors.shape[0] == 0:
            return np.zeros((0, target_dimension))
            
        # Simple PCA projection for demonstration
        cov = np.cov(self.vectors, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        return self.vectors @ eigenvectors[:, -target_dimension:]

class PhenomenologicalTracker:
    """Tracks phenomenological state across sessions"""
    
    def __init__(self, config: dict = None) -> None:
        # Extract parameters from config or use defaults
        self.embedding_dim = config.get('embedding_dim', 768) if config else 768
        self.latent_dim = config.get('latent_dim', 128) if config else 128
        
        # Initialize PCA later when we have enough samples
        self.pca = None
        self.embedding_history = deque(maxlen=1000)
        self.resonance_history = deque(maxlen=1000)
        self.state_dim = self.latent_dim  # Desired state dimension
        self.session_history = {}
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.info("Initialized PhenomenologicalTracker")
        self.current_session = str(uuid.uuid4())  # Start a new session by default
        self.valence_map = {self.current_session: []}  # Map session id to list of resonance vectors
        self.lattice_wrapper = config.get('lattice_wrapper') if config else None

    def calibrate_space(self, embedding_matrix: np.ndarray):
        """Calibrate PCA with initial embeddings"""
        if self.pca is None:
            self.pca = IncrementalPCA(n_components=self.state_dim)
        self.pca.partial_fit(embedding_matrix)
        
    def project_to_semantic_space(self, vector: np.ndarray) -> np.ndarray:
        """Project to calibrated phenomenological space"""
        if self.pca is None:
            # Return zeros if PCA not initialized
            return np.zeros(self.state_dim)
        return self.pca.transform(vector.reshape(1, -1))[0]
    
    def update_resonance(self, coherent_input: np.ndarray, coherence: float) -> np.ndarray:
        """Track phenomenological state and return the resonance vector for the current input"""
        try:
            # Store the original embedding
            self.embedding_history.append(coherent_input)
            
            # If we haven't initialized PCA, check if we have enough samples
            if self.pca is None:
                if len(self.embedding_history) >= 51:
                    # Initialize PCA with the accumulated samples
                    self.pca = IncrementalPCA(n_components=min(51, self.state_dim))
                    self.pca.partial_fit(np.array(self.embedding_history))
                else:
                    # Not enough samples, return a zero vector for now
                    return np.zeros(self.state_dim)
            
            # Update PCA with the new sample
            self.pca.partial_fit(coherent_input.reshape(1, -1))
            
            # Project to state vector
            projected = self.project_to_semantic_space(coherent_input)
            
            # Append to resonance history (state vectors)
            self.resonance_history.append(projected)
            
            # Update valence with coherence score
            self.valence_history.append(coherence)
            
            return projected
        except Exception as e:
            self.logger.error(f"State tracking failed: {str(e)}")
            raise

    def start_new_session(self, session_id: Optional[str] = None) -> str:
        """Start a new session, returns the session id"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        self.current_session = session_id
        self.valence_map[session_id] = []
        return session_id

    def get_current_mood(self, session_id: Optional[str] = None) -> np.ndarray:
        """Compute the current mood as the average of the last 10 resonance vectors in the session"""
        if session_id is None:
            session_id = self.current_session
        vectors = self.valence_map.get(session_id, [])
        if not hasattr(self.pca, 'components_'):
            # Return a zero vector of target dimension if PCA not fitted
            return np.zeros(self.latent_dim)
        if not vectors:
            return np.zeros(1)  # Return a default vector of length 1 if no data
        # Return the average of the last 10 vectors
        last_vectors = vectors[-10:]
        avg_vector = np.mean(last_vectors, axis=0)
        return avg_vector

    def measure_affective_congruence(self, new_state: np.ndarray) -> np.ndarray:
        """Measure affective congruence between new state and resonance history"""
        if len(self.valence_map[self.current_session]) == 0:
            return np.array([0.0])
        
        # Compute cosine similarity against each resonance vector
        similarities = []
        for vec in self.valence_map[self.current_session]:
            sim = cosine_similarity([new_state], [vec])[0][0]
            similarities.append(sim)
        
        return np.array(similarities)

    def track_state(self, state: StateVector) -> StateVector:
        """Track a state vector and update resonance history"""
        # Use original_embeddings if available, else embeddings
        embeddings_to_use = state.original_embeddings if state.original_embeddings.size > 0 else state.embeddings
        coherent_input = embeddings_to_use
        self.update_resonance(coherent_input, state.coherence_score)
        # Store in session history
        self.valence_map[self.current_session].append(coherent_input)
        # Update state with resonance vector
        state.add_metric('resonance_vector', coherent_input.tolist())
        # Return updated state
        return state
