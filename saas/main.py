# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
"""FastAPI micro-service exposing Constraint-Lattice as SaaS.

This is an *initial skeleton*:
• `/health` – liveness probe
• `/govern` – POST prompt+output, returns moderated output (optionally streams)
• `/register-model` – stub for future model registration

The implementation is intentionally minimal so that it can run locally without
DB or auth. Real auth, multi-tenancy, billing, and persistence layers will be
added in follow-up iterations.
"""

from __future__ import annotations

# Configure structured logging & metrics before anything else
import constraint_lattice.engine.telemetry  # noqa: F401 – side-effects

import asyncio
from typing import AsyncGenerator, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Billing / auth packages (keep at import-time to avoid circulars)
from saas.billing import webhooks as billing_webhooks

from saas import get_engine
from saas.deps import rate_limit, get_tenant_policy, enforce_scope

app = FastAPI(title="Constraint-Lattice SaaS", version="0.1.0")

# Register sub-routers ------------------------------------------------------
app.include_router(billing_webhooks.router)


class GovernRequest(BaseModel):
    prompt: str
    output: str
    stream: bool = False  # future-proof flag


class GovernResponse(BaseModel):
    moderated: str


@app.get("/health", tags=["meta"])
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/govern", response_model=GovernResponse, tags=["govern"], dependencies=[Depends(rate_limit), Depends(lambda: enforce_scope("default"))])
async def govern(req: GovernRequest, policy=Depends(get_tenant_policy)):
    engine = get_engine()

    # Inject tenant-specific constraints if policy provided
    engine.constraints = policy.constraints  # noqa: WPS437 workable for per-request

    moderated = engine.run(req.prompt, req.output)

    return GovernResponse(moderated=moderated)


@app.post("/govern/stream", tags=["govern"], dependencies=[Depends(rate_limit), Depends(lambda: enforce_scope("default"))])
async def govern_stream(req: GovernRequest, policy=Depends(get_tenant_policy)):
    if not req.stream:
        raise HTTPException(status_code=400, detail="Set stream=true or call /govern")

    engine = get_engine()
    engine.constraints = policy.constraints

    async def _generator() -> AsyncGenerator[str, None]:
        # Naïve chunking for demo – split by words.
        words = req.output.split()
        partial = ""
        for word in words:
            partial += word + " "
            moderated = engine.run(req.prompt, partial)
            yield moderated
            await asyncio.sleep(0.01)  # Simulate latency

    return StreamingResponse(_generator(), media_type="text/plain")


# ---------------------------------------------------------------------------
# Convenience entry-point for `python -m saas.main` or console-script.
# ---------------------------------------------------------------------------

def run():  # pragma: no cover – simple wrapper
    import uvicorn

    uvicorn.run("saas.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run()
