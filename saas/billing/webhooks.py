# SPDX-License-Identifier: BSL-1.1
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
# Copyright (c) 2025 Lexsight LCC. All rights reserved.
# See saas/LICENSE-BSL.txt for full terms.
"""Stripe webhook endpoint stub.

Implements minimal verification placeholder to keep surface sensible until the
secret key and signature scheme are configured in prod/staging.
"""
from __future__ import annotations

import hmac
import logging
import os
from hashlib import sha256

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/billing", tags=["billing"])


@router.post("/stripe-webhook")
async def stripe_webhook(request: Request):
    # In production we would verify signature using stripe.Signature.
    raw_body = await request.body()
    received_sig = request.headers.get("Stripe-Signature", "")

    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    if not webhook_secret:
        logger.warning("STRIPE_WEBHOOK_SECRET not set; skipping verification!")
    else:
        expected_sig = hmac.new(webhook_secret.encode(), raw_body, sha256).hexdigest()
        if not hmac.compare_digest(expected_sig, received_sig):
            raise HTTPException(status_code=400, detail="Invalid signature")

    logger.info("Received Stripe webhook (%d bytes)", len(raw_body))
    return {"received": True}
