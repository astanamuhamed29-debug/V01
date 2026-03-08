"""Backward-compatibility shim for reply_minimal.

The canonical implementation has moved to :mod:`core.pipeline.template_reply`.
``generate_reply`` is kept here as an alias so that any external callers that
have not yet been updated continue to work without breakage.

.. deprecated::
    Import :func:`core.pipeline.template_reply.render_template_reply` directly.
"""

from __future__ import annotations

from core.pipeline.template_reply import render_template_reply as generate_reply

__all__ = ["generate_reply"]
