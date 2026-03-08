"""Pipeline modules.

Canonical public API
--------------------
* :class:`~core.pipeline.processor.MessageProcessor` — thin OODA orchestrator.
  Call ``await processor.process_message(user_id, text)`` to process a message.
* :class:`~core.pipeline.processor.ProcessResult` — the return type.

Response rendering
------------------
* :func:`~core.pipeline.template_reply.render_template_reply` — LLM-free
  template fallback used by :class:`~core.pipeline.stage_act.ActStage`.
"""

from core.pipeline.processor import MessageProcessor, ProcessResult

__all__ = ["MessageProcessor", "ProcessResult"]
