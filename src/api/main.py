# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
import json
import logging
from typing import Any, Dict, List, Optional

# Third-party imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# Local imports
from constraint_lattice.engine.apply import apply_constraints, AuditStep
from constraint_lattice.constraints.profanity import ProfanityFilter
from constraint_lattice.constraints.length import LengthConstraint
from api.ws import manager

# Initialize FastAPI app
app = FastAPI(
    title="Constraint Lattice API",
    description="API for applying constraints to LLM outputs",
    version="0.1.0"
)

# Configure logging
logger = logging.getLogger(__name__)


class ConstraintRequest(BaseModel):
    """Request model for constraint application.
    
    Attributes:
        text: Text to process
        constraints: List of constraint configurations
    """
    text: str
    constraints: List[Dict[str, Any]]


class ConstraintResponse(BaseModel):
    """Response model for constraint application.
    
    Attributes:
        result: Text after constraint application
        steps: Optional audit trace of applied constraints
    """
    result: str
    steps: Optional[List[Dict[str, Any]]] = None


@app.post("/api/constraints/apply", response_model=ConstraintResponse)
async def apply_constraints_endpoint(request: ConstraintRequest) -> Dict[str, Any]:
    """Apply constraints to input text.
    
    Args:
        request: Constraint application request
        
    Returns:
        Dictionary containing processed text and optional audit trace
        
    Raises:
        HTTPException: For invalid constraint configurations
    """
    try:
        constraint_objs = []
        for constraint_config in request.constraints:
            constraint_type = constraint_config.get("type")
            
            if constraint_type == "profanity":
                constraint_objs.append(ProfanityFilter(
                    replacement=constraint_config.get("replacement", "[FILTERED]")
                ))
            elif constraint_type == "length":
                constraint_objs.append(LengthConstraint(
                    max_length=constraint_config["max"],
                    truncate=constraint_config.get("truncate", True),
                    ellipsis=constraint_config.get("ellipsis", "[...]")
                ))
            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")
        
        # Log the constraint objects
        logger.info(f"Converted {len(constraint_objs)} constraints")
        for i, obj in enumerate(constraint_objs):
            logger.info(f"Constraint {i}: {obj}, type: {type(obj)}")
        
        # Apply constraints
        processed, trace = apply_constraints(
            prompt="",
            output=request.text,
            constraints=constraint_objs,
            return_trace=True
        )
        
        # Broadcast trace updates
        for step in trace:
            await manager.broadcast(json.dumps(step.to_dict()))
        
        # Convert audit trace to serializable format
        audit_trace = [step.to_dict() for step in trace] if trace else None
        
        return {
            "result": processed,
            "steps": audit_trace
        }
        
    except KeyError as e:
        logger.error(f"Missing required parameter: {e}")
        raise HTTPException(status_code=400, detail=f"Missing required parameter: {e}")
    except Exception as e:
        logger.exception("Error applying constraints")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint.
    
    Returns:
        Status message
    """
    return {"status": "healthy"}


@app.websocket("/ws/trace")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time constraint application.
    
    Args:
        websocket: WebSocket connection
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
