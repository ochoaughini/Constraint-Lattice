# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from fastapi import FastAPI
from pydantic import BaseModel

from sdk.engine import ConstraintEngine

app = FastAPI()
engine = ConstraintEngine()


class ConstraintRequest(BaseModel):
    prompt: str
    output: str
    return_trace: bool = False


@app.post("/govern")
def govern(req: ConstraintRequest):
    result = engine.run(req.prompt, req.output, return_trace=req.return_trace)
    if req.return_trace:
        output, trace = result
        return {"output": output, "audit_trace": [t.__dict__ for t in trace]}
    else:
        return {"output": result}
