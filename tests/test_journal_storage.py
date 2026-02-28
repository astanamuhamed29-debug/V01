import asyncio

from core.journal.storage import JournalStorage


def test_journal_append_and_read(tmp_path):
    async def scenario() -> None:
        db_path = tmp_path / "test.db"
        journal = JournalStorage(db_path=db_path)

        await journal.append(user_id="me", timestamp="2026-02-28T10:00:00+00:00", text="Первое сообщение", source="cli")
        await journal.append(user_id="me", timestamp="2026-02-28T10:01:00+00:00", text="Второе сообщение", source="cli")

        entries = await journal.list_entries("me", limit=10)
        assert len(entries) == 2
        assert entries[0].text == "Второе сообщение"
        assert entries[1].text == "Первое сообщение"

    asyncio.run(scenario())
