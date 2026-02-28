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

## Проверочный сценарий

В CLI введите по очереди:

1. `Хочу сделать свою личную ОС SELF-OS.`
2. `Надо выделить вечер, чтобы набросать архитектуру.`
3. `Я боюсь, что не вывезу такой большой проект.`

После этого в БД появятся соответствующие записи `journal_entries`, а в графе — узлы `PROJECT`, `NOTE`, `TASK`, `BELIEF` и связи `OWNS_PROJECT`, `HAS_TASK`, `HOLDS_BELIEF`, `RELATES_TO`.

Для Telegram-сценария сообщение вида `Привет, я хочу переехать` создаёт `PROJECT` с именем `переезд`.
