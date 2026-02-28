"""MemoryScheduler — APScheduler-based cron for memory lifecycle tasks.

Schedules three recurring jobs for each registered user:

* **consolidate** — every 24 hours (cluster low-retention NOTE nodes).
* **abstract**    — every 7 days (LLM-powered belief summarisation).
* **forget**      — every 7 days (prune stale edges/orphan nodes).

The scheduler uses APScheduler's ``AsyncIOScheduler`` and runs
entirely within the existing asyncio event loop — no extra threads.

Usage::

    from core.scheduler.memory_scheduler import MemoryScheduler

    scheduler = MemoryScheduler(storage)
    scheduler.start()       # call once at bot startup
    await scheduler.stop()  # call on graceful shutdown

See FRONTIER_VISION_REPORT §2 — *Scheduler for Memory Lifecycle*.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.memory.consolidator import MemoryConsolidator

if TYPE_CHECKING:
    from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

# Default schedule intervals
CONSOLIDATE_INTERVAL_HOURS = 24
ABSTRACT_INTERVAL_HOURS = 168    # 7 days
FORGET_INTERVAL_HOURS = 168      # 7 days


class MemoryScheduler:
    """Schedules periodic memory lifecycle jobs via APScheduler.

    Parameters
    ----------
    storage:
        The :class:`~core.graph.storage.GraphStorage` instance.
    consolidate_hours:
        Interval (hours) between consolidation runs.
    abstract_hours:
        Interval (hours) between abstraction runs.
    forget_hours:
        Interval (hours) between forgetting runs.
    """

    def __init__(
        self,
        storage: GraphStorage,
        *,
        consolidate_hours: int = CONSOLIDATE_INTERVAL_HOURS,
        abstract_hours: int = ABSTRACT_INTERVAL_HOURS,
        forget_hours: int = FORGET_INTERVAL_HOURS,
    ) -> None:
        self._storage = storage
        self._consolidator = MemoryConsolidator(storage)
        self._consolidate_hours = consolidate_hours
        self._abstract_hours = abstract_hours
        self._forget_hours = forget_hours
        self._scheduler = None  # lazy init to avoid import issues
        self._started = False

    def start(self) -> None:
        """Start the APScheduler with configured jobs."""
        if self._started:
            logger.warning("MemoryScheduler already started")
            return

        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
            from apscheduler.triggers.interval import IntervalTrigger
        except ImportError as exc:
            logger.error(
                "APScheduler is required for MemoryScheduler. "
                "Install with: pip install apscheduler"
            )
            raise ImportError(
                "apscheduler is required. pip install apscheduler"
            ) from exc

        self._scheduler = AsyncIOScheduler()

        self._scheduler.add_job(
            self._run_consolidate,
            trigger=IntervalTrigger(hours=self._consolidate_hours),
            id="memory_consolidate",
            name="Memory Consolidation (daily)",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._run_abstract,
            trigger=IntervalTrigger(hours=self._abstract_hours),
            id="memory_abstract",
            name="Memory Abstraction (weekly)",
            replace_existing=True,
        )
        self._scheduler.add_job(
            self._run_forget,
            trigger=IntervalTrigger(hours=self._forget_hours),
            id="memory_forget",
            name="Memory Forgetting (weekly)",
            replace_existing=True,
        )

        self._scheduler.start()
        self._started = True
        logger.info(
            "MemoryScheduler started: consolidate=%dh, abstract=%dh, forget=%dh",
            self._consolidate_hours,
            self._abstract_hours,
            self._forget_hours,
        )

    async def stop(self) -> None:
        """Shutdown the scheduler gracefully."""
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._started = False
            logger.info("MemoryScheduler stopped")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is currently running."""
        return self._started

    def get_jobs(self) -> list[dict]:
        """Return info about scheduled jobs (for diagnostics)."""
        if self._scheduler is None:
            return []
        return [
            {
                "id": job.id,
                "name": job.name,
                "next_run_time": str(job.next_run_time),
            }
            for job in self._scheduler.get_jobs()
        ]

    # ── Job runners ──────────────────────────────────────────────

    async def _run_consolidate(self) -> None:
        """Run consolidation for all users."""
        logger.info("MemoryScheduler: starting consolidation pass")
        user_ids = await self._storage.get_all_user_ids()
        for user_id in user_ids:
            try:
                report = await self._consolidator.consolidate(user_id)
                if report.clusters_found > 0:
                    logger.info(
                        "Consolidated user=%s: %d clusters, %d merged, %d new",
                        user_id,
                        report.clusters_found,
                        report.nodes_merged,
                        report.new_nodes_created,
                    )
            except Exception as exc:
                logger.warning(
                    "Consolidation failed for user=%s: %s", user_id, exc
                )

    async def _run_abstract(self) -> None:
        """Run abstraction for all users."""
        logger.info("MemoryScheduler: starting abstraction pass")
        user_ids = await self._storage.get_all_user_ids()
        for user_id in user_ids:
            try:
                report = await self._consolidator.abstract(user_id)
                if report.abstracted > 0:
                    logger.info(
                        "Abstracted user=%s: %d candidates, %d abstracted",
                        user_id,
                        report.candidates,
                        report.abstracted,
                    )
            except Exception as exc:
                logger.warning(
                    "Abstraction failed for user=%s: %s", user_id, exc
                )

    async def _run_forget(self) -> None:
        """Run forgetting for all users."""
        logger.info("MemoryScheduler: starting forget pass")
        user_ids = await self._storage.get_all_user_ids()
        for user_id in user_ids:
            try:
                report = await self._consolidator.forget(user_id)
                if report.edges_removed > 0 or report.nodes_tombstoned > 0:
                    logger.info(
                        "Forget user=%s: %d edges removed, %d nodes tombstoned",
                        user_id,
                        report.edges_removed,
                        report.nodes_tombstoned,
                    )
            except Exception as exc:
                logger.warning(
                    "Forgetting failed for user=%s: %s", user_id, exc
                )

    # ── Manual trigger (for testing / CLI) ───────────────────────

    async def run_all_now(self, user_id: str | None = None) -> dict:
        """Run all three phases immediately for one or all users.

        Returns a summary dict with the reports.
        """
        user_ids = [user_id] if user_id else await self._storage.get_all_user_ids()
        results: dict[str, dict] = {}

        for uid in user_ids:
            try:
                c = await self._consolidator.consolidate(uid)
                a = await self._consolidator.abstract(uid)
                f = await self._consolidator.forget(uid)
                results[uid] = {
                    "consolidate": {
                        "clusters": c.clusters_found,
                        "merged": c.nodes_merged,
                        "created": c.new_nodes_created,
                    },
                    "abstract": {
                        "candidates": a.candidates,
                        "abstracted": a.abstracted,
                    },
                    "forget": {
                        "edges_removed": f.edges_removed,
                        "nodes_tombstoned": f.nodes_tombstoned,
                    },
                }
            except Exception as exc:
                results[uid] = {"error": str(exc)}

        return results
