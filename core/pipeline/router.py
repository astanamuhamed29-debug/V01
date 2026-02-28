from __future__ import annotations

import re


INTENTS = {
    "REFLECTION",
    "EVENT_REPORT",
    "IDEA",
    "TASK_LIKE",
    "FEELING_REPORT",
    "META",
}

META_PATTERNS = re.compile(
    r"\b(зачем|в\s+ч[её]м(?:\s+\w+){0,3}\s+смысл|какой\s+смысл|в\s+ч[её]м(?:\s+\w+){0,3}\s+польза|что\s+это\s+да[её]т|"
    r"что\s+это\s+мне\s+да[её]т|нафига|для\s+чего\s+это|зачем\s+это\s+вс[её])\b",
    re.IGNORECASE,
)


def classify(text: str) -> str:
    lowered = text.lower().strip()

    if META_PATTERNS.search(lowered):
        return "META"

    if re.search(r"\b(помощь|help|что ты умеешь|команды)\b", lowered):
        return "META"

    if re.search(r"\b(надо|нужно|сделать|задача|дедлайн|план)\b", lowered):
        return "TASK_LIKE"

    if re.search(r"\b(боюсь|страшно|тревож|груст|злюсь|чувствую|ненавижу\s+себя|стыд|устал|беспомощ|залип)\b", lowered):
        return "FEELING_REPORT"

    if re.search(r"\b(идея|придумал|хочу сделать|можно сделать)\b", lowered):
        return "IDEA"

    if re.search(r"\b(сегодня|вчера|произошло|случилось|было)\b", lowered):
        return "EVENT_REPORT"

    return "REFLECTION"
