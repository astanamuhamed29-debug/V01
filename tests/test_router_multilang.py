from core.pipeline.router import classify


cases = [
    ("тревожусь перед встречей", "FEELING_REPORT"),
    ("переживаю за результат", "FEELING_REPORT"),
    ("нет сил вообще", "FEELING_REPORT"),
    ("бесит эта ситуация", "FEELING_REPORT"),
    ("не забыть отправить файл", "TASK_LIKE"),
    ("нужно закончить до пятницы", "TASK_LIKE"),
    ("запланировал встречу на завтра", "TASK_LIKE"),
    ("хочу попробовать медитацию", "IDEA"),
    ("было бы здорово поехать", "IDEA"),
    ("а что если я начну вести дневник", "IDEA"),
    ("поговорил с другом сегодня", "EVENT_REPORT"),
    ("узнал интересную вещь вчера", "EVENT_REPORT"),
    ("зачем я вообще это делаю", "META"),
    ("что ты умеешь", "META"),
    ("feeling really anxious today", "FEELING_REPORT"),
    ("i'm so tired and burned out", "FEELING_REPORT"),
    ("i feel empty", "FEELING_REPORT"),
    ("need to finish the report", "TASK_LIKE"),
    ("don't forget to call mom", "TASK_LIKE"),
    ("want to try meditation", "IDEA"),
    ("what if i start a blog", "IDEA"),
    ("just had a great meeting", "EVENT_REPORT"),
    ("talked to my friend yesterday", "EVENT_REPORT"),
    ("what can you do", "META"),
    ("why should i use this", "META"),
    ("feeling грусть сегодня", "FEELING_REPORT"),
    ("надо сделать todo list", "TASK_LIKE"),
]


def test_router_coverage():
    failed = []
    for text, expected in cases:
        result = classify(text)
        if result != expected:
            failed.append(f"  '{text}' → got '{result}', expected '{expected}'")
    assert not failed, "Router coverage failures:\n" + "\n".join(failed)


def test_no_silent_failure():
    samples = [
        "asdf",
        "???",
        "ok",
        "да",
        "нет",
        "👍",
        "...",
        "привет",
        "hello",
    ]
    for sample in samples:
        result = classify(sample)
        assert result in {
            "REFLECTION",
            "FEELING_REPORT",
            "TASK_LIKE",
            "IDEA",
            "EVENT_REPORT",
            "META",
            "UNKNOWN",
        }, f"Invalid intent '{result}' for '{sample}'"
