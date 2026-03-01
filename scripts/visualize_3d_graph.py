import asyncio
import json
import os
import sys
import webbrowser
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.graph.storage import GraphStorage

COLORS = {
    "EMOTION": "#ff4b4b",  # Red
    "THOUGHT": "#4b8bff",  # Blue
    "EVENT": "#4bff8b",    # Green
    "VALUE": "#ffd700",    # Gold
    "NEED": "#ff9d00",     # Orange
    "PART": "#b84bff",     # Purple
    "BELIEF": "#ff4bed",   # Pink
    "PERSON": "#4bffff",   # Cyan
    "INSIGHT": "#ffffff",  # White
    "SOMA": "#ffb84b",     # Warm yellow
    "ACTION": "#a0a0a0",   # Gray
    "GOAL": "#ffff00",     # Bright yellow
    "PROJECT": "#00ff00",  # Bright green
    "TASK": "#88ff88",     # Light green
}

async def generate_3d_graph(db_path: str, output_html: str = "graph_3d.html", user_id: str = None):
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    storage = GraphStorage(db_path)
    
    # Get all nodes
    query = "SELECT id, user_id, type, key, text, created_at FROM nodes"
    params = []
    if user_id:
        query += " WHERE user_id = ?"
        params.append(user_id)
        
    async with storage._get_conn() as db:
        async with db.execute(query, params) as cursor:
            nodes_raw = await cursor.fetchall()
            
    # Get all edges
    query_edges = "SELECT source_id, target_id, relation FROM edges"
    async with storage._get_conn() as db:
        async with db.execute(query_edges) as cursor:
            edges_raw = await cursor.fetchall()

    await storage.close()

    if not nodes_raw:
        print("Graph is empty!")
        return

    node_ids = {n[0] for n in nodes_raw}
    
    nodes_data = []
    for row in nodes_raw:
        n_id, u_id, n_type, key, text, created_at = row
        label = text if text else key
        if len(label) > 60:
            label = label[:57] + "..."
            
        nodes_data.append({
            "id": n_id,
            "name": f"[{n_type}] {label}",
            "group": n_type,
            "color": COLORS.get(n_type, "#888888"),
            "val": 1  # Will be updated based on degree
        })

    edges_data = []
    for row in edges_raw:
        src, tgt, rel = row
        if src in node_ids and tgt in node_ids:
            edges_data.append({
                "source": src,
                "target": tgt,
                "name": rel
            })
            
    # Calculate degree for node sizing
    degrees = {n: 0 for n in node_ids}
    for e in edges_data:
        degrees[e["source"]] += 1
        degrees[e["target"]] += 1
        
    for n in nodes_data:
        # Base size + degree scaling
        n["val"] = max(1, (degrees.get(n["id"], 0) * 0.5) + 1)

    graph_data = {"nodes": nodes_data, "links": edges_data}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SELF-OS: 3D Memory Graph</title>
    <style>
        body {{ margin: 0; padding: 0; background-color: #050510; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; }}
        #3d-graph {{ width: 100vw; height: 100vh; }}
        #ui-overlay {{ position: absolute; top: 20px; left: 20px; pointer-events: none; z-index: 10; background: rgba(10,10,25,0.7); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); }}
        h1 {{ margin: 0 0 10px 0; font-size: 24px; letter-spacing: 1px; font-weight: 300; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 14px; opacity: 0.9; }}
        .color-box {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; box-shadow: 0 0 8px currentColor; }}
        .stats {{ margin-top: 20px; font-size: 13px; color: #aaa; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px; }}
    </style>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <script src="https://unpkg.com/three"></script>
</head>
<body>
    <div id="ui-overlay">
        <h1>SELF-OS: IDENTITY & MEMORY</h1>
        <div id="legend"></div>
        <div class="stats">
            Nodes: {len(nodes_data)}<br>
            Edges: {len(edges_data)}
        </div>
    </div>
    <div id="3d-graph"></div>

    <script>
        const gData = {json.dumps(graph_data)};
        const colors = {json.dumps(COLORS)};
        
        // Build legend
        const legend = document.getElementById('legend');
        const usedGroups = new Set(gData.nodes.map(n => n.group));
        
        Array.from(usedGroups).sort().forEach(group => {{
            const color = colors[group] || '#888888';
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<div class="color-box" style="background-color: ${{color}}; color: ${{color}};"></div>${{group}}`;
            legend.appendChild(item);
        }});

        // Initialize Force Graph 3D
        const Graph = ForceGraph3D()(document.getElementById('3d-graph'))
            .graphData(gData)
            .nodeLabel('name')
            .nodeColor('color')
            .nodeVal('val')
            .nodeOpacity(0.9)
            .linkDirectionalArrowLength(3.5)
            .linkDirectionalArrowRelPos(1)
            .linkColor(() => 'rgba(255,255,255,0.2)')
            .linkWidth(0.5)
            .backgroundColor('#050510')
            .onNodeHover(node => document.body.style.cursor = node ? 'pointer' : null);
            
        // Use Bloom effect from Three.js for glowing nodes
        const {{ UnrealBloomPass }} = THREE;
        const bloomPass = new UnrealBloomPass();
        bloomPass.strength = 1.5;
        bloomPass.radius = 0.5;
        bloomPass.threshold = 0.1;
        Graph.postProcessingComposer().addPass(bloomPass);

        // Auto-rotation around the center
        let angle = 0;
        const distance = 400;
        setInterval(() => {{
            angle += Math.PI / 2000;
            Graph.cameraPosition({{
                x: distance * Math.sin(angle),
                z: distance * Math.cos(angle)
            }});
        }}, 10);
    </script>
</body>
</html>
"""
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"✅ Generated 3D interactive graph at: {os.path.abspath(output_html)}")
    
    # Open automatically in browser
    webbrowser.open(f"file://{os.path.abspath(output_html)}")

if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "demo_graph.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "graph_3d.html"
    asyncio.run(generate_3d_graph(db, out))
