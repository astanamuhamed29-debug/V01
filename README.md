# SELF-OS (Stage 1)

Локальное накопительное ядро персонального графа (SELF-Graph) на Python 3.11+.

## Что реализовано

- Приём сообщений через CLI.
- Raw journal в SQLite (`journal_entries`).
- Синхронный pipeline: journal → router → extractors → graph API → reply.
- SELF-Graph в SQLite (`nodes`, `edges`) с upsert по `user_id + type + key`.
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
