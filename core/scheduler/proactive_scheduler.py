"""
ProactiveScheduler — фоновый asyncio-таск.

Запускается один раз при старте бота как asyncio.create_task().
Каждые CHECK_INTERVAL секунд проверяет всех пользователей.
Для каждого пользователя запускает SignalDetector,
который оценивает нужно ли написать сообщение.

Гарантии:
- Не пишет пользователю чаще чем MIN_INTERVAL_HOURS
- Не пишет если пользователь не активен >INACTIVITY_DAYS
- Не пишет если сигнал слабый (score < SIGNAL_THRESHOLD)
- При любой ошибке логирует warning и продолжает следующего
- Graceful shutdown через asyncio.CancelledError
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

if TYPE_CHECKING:
    from aiogram import Bot

    from core.analytics.calibrator import ThresholdCalibrator
    from core.analytics.pattern_analyzer import PatternAnalyzer, PatternReport
    from core.graph.storage import GraphStorage

logger = logging.getLogger(__name__)

from core.defaults import (
    PROACTIVE_CHECK_INTERVAL as CHECK_INTERVAL,
    PROACTIVE_MIN_INTERVAL_HOURS as MIN_INTERVAL_HOURS,
    PROACTIVE_INACTIVITY_DAYS as INACTIVITY_DAYS,
    PROACTIVE_SIGNAL_THRESHOLD as SIGNAL_THRESHOLD,
    PROACTIVE_MIN_DATA_NODES as MIN_DATA_NODES,
    PROACTIVE_SILENCE_BREAK_DAYS,
)


@dataclass(slots=True)
class ProactiveSignal:
    user_id: str
    signal_type: str
    score: float
    message: str
    context: dict


class SignalDetector:
    def detect(self, report: PatternReport) -> list[ProactiveSignal]:
        if not report.has_enough_data:
            return []

        signals: list[ProactiveSignal] = []

        detectors = [
            self._detect_mood_decline,
            self._detect_part_surge,
            self._detect_unmet_need,
            self._detect_distortion_spike,
            self._detect_syndrome,
            self._detect_silence_break,
        ]
        for detector in detectors:
            try:
                result = detector(report)
                if result is not None:
                    signals.append(result)
            except Exception as exc:
                logger.warning("SignalDetector %s failed: %s", detector.__name__, exc)

        signals.sort(key=lambda item: item.score, reverse=True)
        return signals

    def _detect_mood_decline(self, report: PatternReport) -> ProactiveSignal | None:
        score = 0.0
        contexts: dict[str, object] = {}

        high_needs = [need for need in report.need_profile if need.total_signals >= 3]
        if high_needs:
            score += 0.5
            contexts["high_needs"] = [need.need_name for need in high_needs]
            score += min(len(high_needs) * 0.1, 0.2)

        growing_stress_parts = [
            part
            for part in report.part_dynamics
            if part.trend == "growing" and part.subtype in {"exile", "firefighter", "manager", "critic"}
        ]
        if growing_stress_parts:
            score += min(len(growing_stress_parts) * 0.1, 0.2)
            contexts["growing_parts"] = [part.part_name for part in growing_stress_parts]

        if score < SIGNAL_THRESHOLD:
            return None

        score = min(score, 0.9)
        top_need = high_needs[0].need_name if high_needs else "безопасности"
        message = (
            f"Похоже, в последнее время снова поднимается напряжение вокруг потребности в {top_need}. "
            "Хочешь коротко проверить, что сейчас самое сложное?"
        )
        return ProactiveSignal(
            user_id=report.user_id,
            signal_type="mood_decline",
            score=score,
            message=message,
            context=contexts,
        )

    def _detect_part_surge(self, report: PatternReport) -> ProactiveSignal | None:
        growing = [part for part in report.part_dynamics if part.trend == "growing"]
        if not growing:
            return None

        growing.sort(key=lambda item: item.appearances, reverse=True)
        part = growing[0]

        subtype = part.subtype.lower()
        if subtype == "firefighter":
            score = 0.7
            message = (
                f"Замечаю, что {part.part_name} стал появляться чаще. "
                "Похоже, что-то давит. Что сейчас происходит?"
            )
        elif subtype == "critic":
            score = 0.65
            message = (
                f"Кажется, {part.part_name} в последние дни звучит громче. "
                "Как ты к себе сейчас относишься?"
            )
        elif subtype == "exile":
            score = 0.8
            message = (
                f"Чувствую, что {part.part_name} снова рядом. "
                "Если хочешь, можем бережно посмотреть, что именно задело."
            )
        else:
            score = 0.45
            message = (
                f"Вижу, что часть «{part.part_name}» стала активнее. "
                "Хочешь немного распаковать это вместе?"
            )

        return ProactiveSignal(
            user_id=report.user_id,
            signal_type="part_surge",
            score=score,
            message=message,
            context={
                "part_name": part.part_name,
                "part_subtype": part.subtype,
                "appearances": part.appearances,
            },
        )

    def _detect_unmet_need(self, report: PatternReport) -> ProactiveSignal | None:
        if not report.need_profile:
            return None

        top_need = max(report.need_profile, key=lambda item: item.total_signals)
        if top_need.total_signals < 4:
            return None

        score = min(0.5 + top_need.total_signals * 0.05, 0.85)
        message = (
            f"Я замечаю, что потребность в {top_need.need_name} сигнализирует уже несколько раз. "
            "Есть что-то, что тревожит прямо сейчас?"
        )
        return ProactiveSignal(
            user_id=report.user_id,
            signal_type="need_unmet",
            score=score,
            message=message,
            context={"need": top_need.need_name, "signals": top_need.total_signals},
        )

    def _detect_distortion_spike(self, report: PatternReport) -> ProactiveSignal | None:
        if not report.cognition_patterns:
            return None

        top = max(report.cognition_patterns, key=lambda item: item.count)
        if top.count < 4:
            return None

        score = min(0.4 + top.count * 0.05, 0.75)
        message = (
            f"Замечаю повторяющийся мыслительный паттерн: {top.distortion_ru}. "
            "Это нормально — психика так пытается защитить. Хочешь посмотреть на это вместе?"
        )
        return ProactiveSignal(
            user_id=report.user_id,
            signal_type="distortion_spike",
            score=score,
            message=message,
            context={"distortion": top.distortion, "count": top.count, "example": top.example_thought},
        )

    def _detect_silence_break(self, report: PatternReport) -> ProactiveSignal | None:
        if not report.last_activity_at:
            return None
        last_activity = _parse_iso(report.last_activity_at)
        if datetime.now(timezone.utc) - last_activity < timedelta(days=PROACTIVE_SILENCE_BREAK_DAYS):
            return None

        return ProactiveSignal(
            user_id=report.user_id,
            signal_type="silence_break",
            score=0.42,
            message="Давно не виделись. Как ты сейчас?",
            context={"last_activity_at": report.last_activity_at},
        )

    def _detect_syndrome(self, report: PatternReport) -> ProactiveSignal | None:
        if not report.syndromes:
            return None

        top_syndrome = report.syndromes[0]
        if top_syndrome.score < 0.6:
            return None

        names = [name for name in top_syndrome.nodes if len(name) < 20][:3]
        if len(names) < 2:
            return None

        message = (
            "Я анализировал структуру твоих записей и заметил устойчивый узел. "
            f"Каждый раз, когда сходятся {', '.join(names)} — возникает сильное напряжение. "
            "Как думаешь, что их связывает?"
        )

        return ProactiveSignal(
            user_id=report.user_id,
            signal_type="syndrome_detected",
            score=0.75,
            message=message,
            context={"nodes": top_syndrome.nodes},
        )


class ProactiveScheduler:
    def __init__(
        self,
        bot: "Bot",
        storage: "GraphStorage",
        analyzer: "PatternAnalyzer",
        calibrator: "ThresholdCalibrator | None" = None,
        check_interval: int = CHECK_INTERVAL,
    ) -> None:
        self.bot = bot
        self.storage = storage
        self.analyzer = analyzer
        self._calibrator = calibrator
        self.check_interval = check_interval
        self._detector = SignalDetector()
        self._task: asyncio.Task | None = None
        self._calibrated_users: set[str] = set()

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            logger.warning("ProactiveScheduler already running")
            return
        self._task = asyncio.create_task(self._loop(), name="proactive_scheduler")
        logger.info("ProactiveScheduler started (interval=%ds)", self.check_interval)

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ProactiveScheduler stopped")

    async def _loop(self) -> None:
        logger.info("ProactiveScheduler loop started")
        while True:
            try:
                await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("ProactiveScheduler loop error: %s", exc, exc_info=True)
            await asyncio.sleep(self.check_interval)

    async def _run_once(self) -> None:
        user_ids = await self.storage.get_all_user_ids()
        logger.debug("ProactiveScheduler checking %d users", len(user_ids))

        now = datetime.now(timezone.utc)
        for user_id in user_ids:
            try:
                await self._check_user(user_id, now)
            except Exception as exc:
                logger.warning("ProactiveScheduler failed for user=%s: %s", user_id, exc)

    async def _check_user(self, user_id: str, now: datetime) -> None:
        if self._calibrator and user_id not in self._calibrated_users:
            try:
                await self._calibrator.load(user_id)
                self._calibrated_users.add(user_id)
            except Exception as exc:
                logger.warning("Calibrator load failed for user=%s: %s", user_id, exc)

        last_activity = await self.storage.get_last_activity_at(user_id)
        if last_activity is None:
            return

        last_activity_dt = _parse_iso(last_activity)
        if (now - last_activity_dt).days > INACTIVITY_DAYS:
            logger.debug("user=%s inactive >%d days, skip", user_id, INACTIVITY_DAYS)
            return

        state = await self.storage.get_scheduler_state(user_id)
        if state and state.get("last_proactive_at"):
            last_sent_dt = _parse_iso(state["last_proactive_at"])
            hours_since = (now - last_sent_dt).total_seconds() / 3600
            if hours_since < MIN_INTERVAL_HOURS:
                logger.debug(
                    "user=%s cooldown (%.1fh < %dh), skip",
                    user_id,
                    hours_since,
                    MIN_INTERVAL_HOURS,
                )
                return

        report = await self.analyzer.analyze(user_id, days=30)
        if not report.has_enough_data:
            await self.storage.upsert_scheduler_state(user_id, last_checked_at=now.isoformat())
            return

        signals = self._detector.detect(report)
        best: ProactiveSignal | None = None
        for signal in signals:
            threshold = self._calibrator.get_threshold(signal.signal_type) if self._calibrator else SIGNAL_THRESHOLD
            if signal.score >= threshold:
                best = signal
                break

        now_iso = now.isoformat()
        if best is None:
            await self.storage.upsert_scheduler_state(user_id, last_checked_at=now_iso)
            logger.debug("user=%s no signal above threshold", user_id)
            return

        try:
            await self.bot.send_message(
                chat_id=int(user_id),
                text=best.message,
                reply_markup=_make_feedback_keyboard(best.signal_type, best.score),
            )
            logger.info(
                "Proactive message sent to user=%s type=%s score=%.2f",
                user_id,
                best.signal_type,
                best.score,
            )
            await self.storage.upsert_scheduler_state(
                user_id,
                last_proactive_at=now_iso,
                last_checked_at=now_iso,
                increment_sent=True,
            )
        except Exception as exc:
            logger.warning("Failed to send proactive to user=%s: %s", user_id, exc)
            await self.storage.upsert_scheduler_state(user_id, last_checked_at=now_iso)


def _parse_iso(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _make_feedback_keyboard(signal_type: str, score: float) -> InlineKeyboardMarkup:
    ts = int(time.time())
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Точно 👍",
                    callback_data=f"fb:1:{signal_type}:{score:.2f}:{ts}",
                ),
                InlineKeyboardButton(
                    text="Не очень 👎",
                    callback_data=f"fb:0:{signal_type}:{score:.2f}:{ts}",
                ),
            ]
        ]
    )
