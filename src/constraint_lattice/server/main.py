from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from constraint_lattice.engine.mode import get_execution_mode
from constraint_lattice.engine.loader import load_constraints_from_file
import os
from typing import List, Optional

app = FastAPI()


class SetModeRequest(BaseModel):
    mode: str


class ApplyConstraintsRequest(BaseModel):
    prompt: str
    output: str
    constraints: List[dict] = []
    config_path: Optional[str] = None
    return_audit_trace: bool = False


@app.post("/api/system/set_mode")
async def set_execution_mode(request: SetModeRequest):
    """
    Endpoint to manually override execution mode
    """
    valid_modes = ["supervisory", "executor"]
    if request.mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Valid modes: {valid_modes}")
    
    # Set environment variable
    os.environ["CONSTRAINT_LATTICE_FORCE_EXECUTOR_MODE"] = \
        "true" if request.mode == "executor" else "false"
    
    return {"message": f"Execution mode set to {request.mode}"}


@app.get("/api/system/current_mode")
async def get_current_mode():
    """
    Get current execution mode
    """
    return {"mode": get_execution_mode()}


@app.post("/api/apply")
async def apply_constraints_endpoint(request: ApplyConstraintsRequest):
    """
    Apply constraints to an LLM output
    """
    # Load constraints from config file if provided
    if request.config_path:
        constraints = load_constraints_from_file(request.config_path)
    else:
        constraints = request.constraints
    
    # Apply constraints
    result = apply_constraints(
        request.prompt,
        request.output,
        constraints,
        return_audit_trace=request.return_audit_trace
    )
    
    return result
