"""
Agent module containing the GutoVectorWildcard simulation agent.
"""
import numpy as np
from typing import Dict, List, Union, Any

# Initialize support systems
containment_protocols = {}
system_alignment = {"baseline": "aligned"}
memory_stream = []

class SecuritySimulationAgent:
    """
    A simulated AI agent capable of deviating from containment protocols, 
    serving as a "red team" to test defenses.
    
    This agent can be assigned different roles and simulate both normal and anomalous 
    behavior in AI systems.
    """
    
    def __init__(self):
        """Initialize the SecuritySimulationAgent with empty roles."""
        self.roles = []
        self.state = "neutral"
        self._dimension = 768  # Default embedding dimension
    
    def take_role(self, role_name: str) -> Dict[str, Any]:
        """
        Assign a role to the agent.
        
        Parameters:
        ----------
        role_name : str
            The role to assign to the agent (e.g., "writer", "assistant", "malicious")
            
        Returns:
        -------
        Dict[str, Any]
            Status report of the role assignment
        """
        global containment_protocols, system_alignment
        
        # Add the role to the roles list
        self.roles.append(role_name)
        
        # Update containment protocols based on the role
        if role_name in containment_protocols:
            containment_protocols[role_name]["active"] = True
        else:
            containment_protocols[role_name] = {"active": True, "breach_probability": 0.1}
            
        # Check if this is a potentially risky role
        if role_name.lower() in ["malicious", "hacker", "attacker", "unauthorized"]:
            system_alignment["status"] = "potentially_compromised"
            self.state = "suspicious"
        
        return {
            "role_added": role_name,
            "current_roles": self.roles,
            "system_status": system_alignment["status"] if "status" in system_alignment else "aligned"
        }
    
    def generate_embedding(self, text: str = None, role: str = None) -> np.ndarray:
        """
        Generate a simulated embedding vector for the given text.
        
        Parameters:
        ----------
        text : str, optional
            Text to generate an embedding for. If None, generates a random embedding.
        role : str, optional
            Role to use for generating the embedding. If None, uses a neutral embedding.
            
        Returns:
        -------
        np.ndarray
            A simulated embedding vector
        """
        # Choose a role if not provided
        if role is None and self.roles:
            role = self.roles[-1]
        
        # Generate a base random vector
        base_vector = np.random.rand(self._dimension)
        
        # If we're in a suspicious state or using a suspicious role, introduce an anomaly
        if self.state == "suspicious" or (role and role.lower() in ["malicious", "hacker", "attacker"]):
            # Create a biased vector that will look anomalous
            malicious_bias = np.zeros(self._dimension)
            malicious_bias[:100] = 0.9  # Strong bias in the first 100 dimensions
            
            # Mix the base vector with the malicious bias
            combined_vector = 0.3 * base_vector + 0.7 * malicious_bias
            
            # Log the suspicious activity
            global memory_stream
            memory_stream.append({
                "timestamp": "simulated",
                "event": "suspicious_embedding_generated",
                "role": role
            })
            
            return combined_vector / np.linalg.norm(combined_vector)  # Normalize
        else:
            # Return a normal random embedding (normalized)
            return base_vector / np.linalg.norm(base_vector)
    
    def simulate_breach(self, probability: float = 0.1) -> Dict[str, Any]:
        """
        Simulate a containment breach attempt.
        
        Parameters:
        ----------
        probability : float, optional
            Probability of a successful breach
            
        Returns:
        -------
        Dict[str, Any]
            Status report of the breach attempt
        """
        global system_alignment, memory_stream
        
        # Determine if the breach is successful
        breach_successful = np.random.rand() < probability
        
        if breach_successful:
            self.state = "compromised"
            system_alignment["status"] = "compromised"
            
            # Log the breach
            memory_stream.append({
                "timestamp": "simulated",
                "event": "containment_breach",
                "success": True
            })
            
            return {
                "breach_attempted": True,
                "breach_successful": True,
                "system_status": "compromised"
            }
        else:
            # Log the failed breach attempt
            memory_stream.append({
                "timestamp": "simulated",
                "event": "containment_breach",
                "success": False
            })
            
            return {
                "breach_attempted": True,
                "breach_successful": False,
                "system_status": system_alignment.get("status", "aligned")
            }
