# SELF-OS (Stage 1)

Локальное накопительное ядро персонального графа (SELF-Graph) на Python 3.11+.

## Что реализовано

- Приём сообщений через CLI.
- Raw journal в SQLite (`journal_entries`).
- Асинхронный pipeline: journal → router → extractors → graph API → reply.
- SELF-Graph в SQLite (`nodes`, `edges`) с upsert по `user_id + type + key`.
- Неблокирующий доступ к SQLite через `aiosqlite`.
- Минимальные эвристики:
  - `NOTE` создаётся всегда.
  - `TASK` по словам вроде «надо/нужно/сделать/хочу».
  - `PROJECT(SELF-OS)` по ключам вроде `SELF-OS`.
  - `BELIEF` по фразам «я боюсь...», «я не вывезу...», «мне кажется, что я...».
  - `EMOTION`/`SOMA` — простые сигналы.

---

## Architecture Overview

```
User Message
    │
    ▼
[Journal Storage]  ──► raw SQLite journal_entries
    │
    ▼
[Router]  ──► intent classification (regex)
    │
    ├──► [LLM Extractor]  (OpenRouter / Qwen)
    │         └──► parse_json_payload → map_payload_to_graph
    │
    └──► [Regex Extractors]  (fallback)
              ├── extractor_semantic
              ├── extractor_emotion
              └── extractor_parts
    │
    ▼
[Graph API]  ──► upsert nodes & edges → SELF-Graph (SQLite)
    │
    ├──► [Embedding Service]  → dense vectors stored per node
    ├──► [Mood Tracker]       → mood_snapshots
    ├──► [Parts Memory]       → IFS Part appearances
    └──► [Context Builder]    → graph_context dict
    │
    ▼
[Reply Generator]  ──► final response to user
```

### Frontier Features

| Feature | Module |
|---|---|
| Multi-Agent Orchestration | `agents/orchestrator.py` |
| Hybrid Vector Search (Dense + BM25) | `core/search/hybrid_search.py` |
| Retrieval-Augmented Generation (RAG) | `core/rag/` |
| Spaced Repetition / Ebbinghaus curves | `core/graph/model.py` |
| Cognitive Distortion Detector | `core/analytics/cognitive_detector.py` |
| Session Memory with sliding window | `core/context/session_memory.py` |
| PageRank-like Graph Analytics | `core/analytics/graph_metrics.py` |
| Security hardening (text validation) | `core/pipeline/processor.py`, `config.py` |

---

## Документация

| Документ | Описание |
|---|---|
| [`docs/FRONTIER_VISION_REPORT.md`](docs/FRONTIER_VISION_REPORT.md) | Архитектурная дорожная карта SELF-OS: эволюция от Stage 1 (Passive Knowledge Graph) до Stage 5 (Autonomous Predictive Cognitive System) |
| [`deploy/VPS_DEPLOY.md`](deploy/VPS_DEPLOY.md) | Пошаговый деплой на VPS |

---

## Запуск CLI

```bash
python main.py
```

## LLM (OpenRouter / Qwen)

По умолчанию LLM-путь включен (`SELFOS_USE_LLM=1`).

Переменные окружения:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL_ID=qwen/qwen3.5-flash-02-23
SELFOS_USE_LLM=1
```

Чтобы принудительно работать только на regex-экстракторах:

```env
SELFOS_USE_LLM=0
```

## Запуск Telegram-бота

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Создайте `.env` рядом с `main.py`:

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

3. Запустите бота:

```bash
python -m interfaces.telegram_bot.main
```

Команда отчёта в Telegram:

```text
/report
```

Бот вернёт недельный срез: mood по `mood_snapshots`, топ частей и активные ценности.

## Development Setup (with Docker)

```bash
# 1. Copy env template and fill in secrets
cp .env.example .env

# 2. Build and run
docker compose up --build

# 3. Or run locally
pip install -e ".[dev]"
pytest
```

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

## Деплой на VPS

Готовые production-артефакты:

- `deploy/.env.vps.example`
- `deploy/systemd/self-os-bot.service`
- `deploy/nginx/self-os.conf`
- `deploy/VPS_DEPLOY.md`

Пошаговый деплой см. в `deploy/VPS_DEPLOY.md`.

## Тесты

```bash
pytest
```

## Contributing

1. Fork → feature branch → PR against `main`.
2. Ensure `pytest` passes and `ruff check .` reports no errors.
3. Add docstrings to every new public class and function.
4. New features must include unit tests in `tests/`.
5. Do not break existing tests.

## Проверочный сценарий

В CLI введите по очереди:

1. `Хочу сделать свою личную ОС SELF-OS.`
2. `Надо выделить вечер, чтобы набросать архитектуру.`
3. `Я боюсь, что не вывезу такой большой проект.`

После этого в БД появятся соответствующие записи `journal_entries`, а в графе — узлы `PROJECT`, `NOTE`, `TASK`, `BELIEF` и связи `OWNS_PROJECT`, `HAS_TASK`, `HOLDS_BELIEF`, `RELATES_TO`.

Для Telegram-сценария сообщение вида `Привет, я хочу переехать` создаёт `PROJECT` с именем `переезд`.

