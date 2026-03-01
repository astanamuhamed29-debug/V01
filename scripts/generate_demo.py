import asyncio
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

from core.graph.api import GraphAPI
from core.graph.storage import GraphStorage
from interfaces.processor_factory import build_processor

# Load the synthetic dialogue
from tests.test_simulation_dialogue import DIALOGUE

load_dotenv()

_EMOTION_LEXICON: list[tuple[str, str, float, float, str]] = [
    ("трев", "тревога", -0.75, 0.75, "безопасность"),
    ("страх", "страх", -0.85, 0.85, "безопасность"),
    ("стыд", "стыд", -0.75, 0.35, "принятие"),
    ("груст", "грусть", -0.55, -0.35, "поддержка"),
    ("печал", "печаль", -0.65, -0.40, "поддержка"),
    ("злост", "злость", -0.50, 0.80, "границы"),
    ("радост", "радость", 0.75, 0.45, "признание"),
    ("горд", "гордость", 0.65, 0.35, "признание"),
    ("облегч", "облегчение", 0.35, -0.25, "безопасность"),
    ("спокой", "спокойствие", 0.45, -0.45, "стабильность"),
    ("устал", "усталость", -0.35, -0.65, "восстановление"),
]


def _extract_demo_signals(text: str) -> list[tuple[str, float, float, str]]:
    lowered = text.lower()
    signals: list[tuple[str, float, float, str]] = []
    for token, label, valence, arousal, need in _EMOTION_LEXICON:
        if token in lowered:
            signals.append((label, valence, arousal, need))
    if not signals:
        signals.append(("тревога", -0.6, 0.6, "безопасность"))
    return signals[:2]


async def _safe_process(
    processor,
    *,
    user_id: str,
    text: str,
    timestamp: str,
    retries: int = 3,
    timeout_seconds: float = 45.0,
) -> bool:
    for attempt in range(1, retries + 1):
        try:
            await asyncio.wait_for(
                processor.process(user_id=user_id, text=text, source="cli", timestamp=timestamp),
                timeout=timeout_seconds,
            )
            return True
        except (TimeoutError, ConnectionError, OSError, asyncio.CancelledError, asyncio.TimeoutError) as exc:
            if attempt == retries:
                print(f"  ✗ Skip message after {retries} retries: {type(exc).__name__}: {exc!r}")
                return False
            backoff = attempt * 1.5
            print(f"  ! Retry {attempt}/{retries} due to: {type(exc).__name__}: {exc!r}. Sleep {backoff:.1f}s")
            await asyncio.sleep(backoff)
        except Exception as exc:
            print(f"  ✗ Unexpected error, skip message: {type(exc).__name__}: {exc!r}")
            return False
    return False


async def _populate_offline_demo(db_path: str, *, limit: int = 10) -> None:
    storage = GraphStorage(db_path)
    api = GraphAPI(storage)
    user_id = "me"
    person = await api.ensure_person_node(user_id)

    selected_dialogue = DIALOGUE[: max(1, min(limit, len(DIALOGUE)))]
    for idx, (_, text) in enumerate(selected_dialogue, 1):
        event = await api.create_node(
            user_id=user_id,
            node_type="EVENT",
            name=(text[:64] + "...") if len(text) > 64 else text,
            key=f"event:demo:{idx}",
        )
        await api.create_edge(
            user_id=user_id,
            source_node_id=person.id,
            target_node_id=event.id,
            relation="DESCRIBES_EVENT",
        )

        for emotion_idx, (label, valence, arousal, need_name) in enumerate(_extract_demo_signals(text), 1):
            emotion = await api.create_node(
                user_id=user_id,
                node_type="EMOTION",
                key=f"emotion:{label}:demo:{idx}:{emotion_idx}",
                metadata={
                    "label": label,
                    "confidence": 0.72,
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": -0.2 if valence < 0 else 0.3,
                    "intensity": 0.75,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            need = await api.find_or_create_node(
                user_id=user_id,
                node_type="NEED",
                key=f"need:{need_name}",
                name=need_name,
                text=f"Эмоция {label} сигнализирует потребность: {need_name}",
            )
            await api.create_edge(
                user_id=user_id,
                source_node_id=person.id,
                target_node_id=emotion.id,
                relation="FEELS",
            )
            await api.create_edge(
                user_id=user_id,
                source_node_id=emotion.id,
                target_node_id=need.id,
                relation="SIGNALS_NEED",
            )

    await storage.close()


async def populate_and_visualize(
    *,
    limit: int = 10,
    timeout_seconds: float = 45.0,
    retries: int = 3,
    offline: bool = False,
):
    db_path = "demo_graph_3d.db"
    
    # Clean previous demo DB if exists
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            db_path = f"demo_graph_3d_{suffix}.db"
            print(f"Base demo DB is locked, using: {db_path}")
        
    print(f"Creating populated database at {db_path}...")
    
    if offline:
        print(f"\nOffline mode enabled. Building deterministic demo graph ({limit} messages)...")
        await _populate_offline_demo(db_path, limit=limit)
    else:
        processor = build_processor(db_path, background_mode=False)

        user_id = "me"
        now = datetime.now(timezone.utc)
        selected_dialogue = DIALOGUE[: max(1, min(limit, len(DIALOGUE)))]

        ok_count = 0
        print(f"\nProcessing {len(selected_dialogue)} synthetic messages to build the graph...")
        print("-" * 50)
        for i, (days_ago, text) in enumerate(selected_dialogue, 1):
            ts = (now - timedelta(days=days_ago, hours=i % 6)).isoformat()
            print(f"[{i:2d}/{len(selected_dialogue)}] Processing: {text[:50]}...")
            ok = await _safe_process(
                processor,
                user_id=user_id,
                text=text,
                timestamp=ts,
                retries=retries,
                timeout_seconds=timeout_seconds,
            )
            ok_count += 1 if ok else 0

        await processor.graph_api.storage.close()

        if ok_count == 0:
            print("No messages were processed successfully online. Falling back to offline demo graph.")
            await _populate_offline_demo(db_path, limit=limit)
    
    print("-" * 50)
    print("Database populated successfully!")
    
    # Now run the visualizer
    from scripts.visualize_3d_graph import generate_3d_graph
    
    print("Generating 3D Visualization...")
    await generate_3d_graph(db_path=db_path, output_html="demo_3d_graph.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo graph and 3D visualization")
    parser.add_argument("limit", nargs="?", type=int, default=10, help="Number of synthetic messages to process")
    parser.add_argument("--offline", action="store_true", help="Build deterministic offline graph (no LLM/API)")
    parser.add_argument("--timeout", type=float, default=45.0, help="Per-message timeout in seconds for online mode")
    parser.add_argument("--retries", type=int, default=3, help="Retries per message for online mode")
    args = parser.parse_args()

    offline_mode = args.offline or not bool(os.getenv("OPENROUTER_API_KEY"))
    if offline_mode and not args.offline:
        print("OPENROUTER_API_KEY not found. Switching to offline demo mode.")

    asyncio.run(
        populate_and_visualize(
            limit=max(1, args.limit),
            timeout_seconds=max(5.0, args.timeout),
            retries=max(1, args.retries),
            offline=offline_mode,
        )
    )
