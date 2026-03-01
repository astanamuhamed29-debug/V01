import asyncio
import os
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv

from core.graph.storage import GraphStorage
from interfaces.processor_factory import build_processor

# Load the synthetic dialogue
from tests.test_simulation_dialogue import DIALOGUE

load_dotenv()

async def populate_and_visualize():
    db_path = "demo_graph_3d.db"
    
    # Clean previous demo DB if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"Creating populated database at {db_path}...")
    
    class _NoopQdrant:
        def upsert_embeddings_batch(self, points): pass
        def search_similar(self, *args, **kwargs): return []
    
    # Must run processor in sync mode to collect nodes sequentially
    processor = build_processor(db_path, background_mode=False)
    processor.qdrant = _NoopQdrant()  # Bypass vector DB to avoid timeouts/errors during demo
    processor._orient.qdrant = processor.qdrant
    processor._act.qdrant = processor.qdrant
    
    user_id = "me"
    
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    
    print("\nProcessing 10 synthetic messages to build the graph...")
    print("-" * 50)
    for i, (days_ago, text) in enumerate(DIALOGUE, 1):
        ts = (now - timedelta(days=days_ago, hours=i % 6)).isoformat()
        print(f"[{i:2d}/10] Processing: {text[:50]}...")
        await processor.process(user_id=user_id, text=text, source="cli", timestamp=ts)
        
    await processor.graph_api.storage.close()
    
    print("-" * 50)
    print("Database populated successfully!")
    
    # Now run the visualizer
    from scripts.visualize_3d_graph import generate_3d_graph
    
    print("Generating 3D Visualization...")
    await generate_3d_graph(db_path=db_path, output_html="demo_3d_graph.html")

if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY must be set in .env")
        exit(1)
        
    asyncio.run(populate_and_visualize())
