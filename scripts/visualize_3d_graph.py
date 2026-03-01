import asyncio
import json
import os
import sys
import webbrowser
from datetime import datetime

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
        
    db = await storage._get_conn()
    async with db.execute(query, params) as cursor:
        nodes_raw = await cursor.fetchall()
            
    # Get all edges
    query_edges = "SELECT source_node_id, target_node_id, relation FROM edges"
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

    # Build emotional trajectory data (3D): x=time-index, y=valence(heuristic), z=arousal(heuristic)
    emotion_rows = [row for row in nodes_raw if row[2] == "EMOTION"]
    emotion_rows.sort(key=lambda row: row[5] or "")

    vad_heuristics = {
        "радость": (0.75, 0.45),
        "гордость": (0.65, 0.35),
        "спокойствие": (0.45, -0.45),
        "облегчение": (0.35, -0.25),
        "тревога": (-0.65, 0.75),
        "страх": (-0.75, 0.85),
        "стыд": (-0.7, 0.35),
        "печаль": (-0.65, -0.4),
        "грусть": (-0.55, -0.35),
        "злость": (-0.5, 0.8),
        "беспомощность": (-0.8, 0.15),
        "усталость": (-0.35, -0.65),
    }

    def _extract_emotion_label(key: str) -> str:
        if not key:
            return "unknown"
        if key.startswith("emotion:"):
            parts = key.split(":")
            if len(parts) >= 2:
                return parts[1].lower()
        return key.lower()

    emotions_data = []
    for idx, row in enumerate(emotion_rows):
        _, _, _, key, text, created_at = row
        label = (text or _extract_emotion_label(key)).lower()
        valence, arousal = vad_heuristics.get(label, (0.0, 0.0))
        try:
            ts = datetime.fromisoformat((created_at or "").replace("Z", "+00:00")).isoformat()
        except ValueError:
            ts = created_at or ""
        emotions_data.append(
            {
                "i": idx,
                "t": ts,
                "label": label,
                "valence": valence,
                "arousal": arousal,
                "color": COLORS.get("EMOTION", "#ff4b4b"),
            }
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SELF-OS: 3D Memory + Emotion State</title>
    <style>
        body {{ margin: 0; padding: 0; background-color: #050510; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; }}
        #toolbar {{ position: absolute; top: 16px; right: 16px; z-index: 12; display: flex; gap: 8px; }}
        .btn {{ background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.25); color: #fff; padding: 8px 12px; border-radius: 8px; cursor: pointer; }}
        .btn.active {{ background: rgba(75,139,255,0.35); border-color: rgba(75,139,255,0.8); }}
        #memory-3d, #emotion-3d {{ width: 100vw; height: 100vh; display: none; }}
        #memory-3d.active, #emotion-3d.active {{ display: block; }}
        #ui-overlay {{ position: absolute; top: 20px; left: 20px; pointer-events: none; z-index: 10; background: rgba(10,10,25,0.7); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); backdrop-filter: blur(10px); max-width: 420px; }}
        h1 {{ margin: 0 0 10px 0; font-size: 24px; letter-spacing: 1px; font-weight: 300; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 14px; opacity: 0.9; }}
        .color-box {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; box-shadow: 0 0 8px currentColor; }}
        .stats {{ margin-top: 20px; font-size: 13px; color: #aaa; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 15px; }}
    </style>
    <script src="https://unpkg.com/3d-force-graph"></script>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
    <div id="toolbar">
        <button id="btn-memory" class="btn active">Memory Graph 3D</button>
        <button id="btn-emotion" class="btn">Emotion State 3D</button>
    </div>
    <div id="ui-overlay">
        <h1>SELF-OS: 3D Visualizer</h1>
        <div id="legend"></div>
        <div class="stats">
            Nodes: {len(nodes_data)}<br>
            Edges: {len(edges_data)}<br>
            Emotions: {len(emotions_data)}
        </div>
    </div>
    <div id="memory-3d" class="active"></div>
    <div id="emotion-3d"></div>

    <script>
        const gData = {json.dumps(graph_data)};
        const emotionData = {json.dumps(emotions_data)};
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

        // Initialize Force Graph 3D (memory)
        const Graph = ForceGraph3D()(document.getElementById('memory-3d'))
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

        // Emotional state 3D (Plotly)
        const emotionX = emotionData.map(e => e.i);
        const emotionY = emotionData.map(e => e.valence);
        const emotionZ = emotionData.map(e => e.arousal);
        const emotionText = emotionData.map(e => `${{e.label}}<br>${{e.t}}`);

        const emotionTrace = {{
            type: 'scatter3d',
            mode: 'lines+markers',
            x: emotionX,
            y: emotionY,
            z: emotionZ,
            text: emotionText,
            hovertemplate: 'idx=%{{x}}<br>valence=%{{y:.2f}}<br>arousal=%{{z:.2f}}<br>%{{text}}<extra></extra>',
            line: {{ color: '#ff8888', width: 6 }},
            marker: {{ size: 5, color: '#ff4b4b' }},
            name: 'Emotion Trajectory'
        }};

        const emotionLayout = {{
            paper_bgcolor: '#050510',
            plot_bgcolor: '#050510',
            font: {{ color: '#ffffff' }},
            margin: {{ l: 0, r: 0, b: 0, t: 0 }},
            scene: {{
                xaxis: {{ title: 'Time Index' }},
                yaxis: {{ title: 'Valence', range: [-1, 1] }},
                zaxis: {{ title: 'Arousal', range: [-1, 1] }},
                bgcolor: '#050510'
            }}
        }};

        Plotly.newPlot('emotion-3d', [emotionTrace], emotionLayout, {{ responsive: true }});

        // Animate emotion timeline cursor
        if (emotionData.length > 0) {{
            let emotionFrame = 1;
            setInterval(() => {{
                emotionFrame = (emotionFrame % emotionData.length) + 1;
                Plotly.restyle('emotion-3d', {{
                    x: [emotionX.slice(0, emotionFrame)],
                    y: [emotionY.slice(0, emotionFrame)],
                    z: [emotionZ.slice(0, emotionFrame)],
                    text: [emotionText.slice(0, emotionFrame)]
                }});
            }}, 700);
        }}

        // View switching
        const btnMemory = document.getElementById('btn-memory');
        const btnEmotion = document.getElementById('btn-emotion');
        const memoryView = document.getElementById('memory-3d');
        const emotionView = document.getElementById('emotion-3d');

        btnMemory.addEventListener('click', () => {{
            btnMemory.classList.add('active');
            btnEmotion.classList.remove('active');
            memoryView.classList.add('active');
            emotionView.classList.remove('active');
        }});

        btnEmotion.addEventListener('click', () => {{
            btnEmotion.classList.add('active');
            btnMemory.classList.remove('active');
            emotionView.classList.add('active');
            memoryView.classList.remove('active');
        }});
    </script>
</body>
</html>
"""
    
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
        
    print(f"✅ Generated 3D memory+emotion visualization at: {os.path.abspath(output_html)}")
    
    # Open automatically in browser
    webbrowser.open(f"file://{os.path.abspath(output_html)}")

if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "demo_graph.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "graph_3d.html"
    asyncio.run(generate_3d_graph(db, out))
