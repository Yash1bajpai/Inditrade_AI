"""
IndiTrade AI - Graph Memory Visualization Exporter
Reads epistemic graph memory (`.agents/graph_memory.sqlite`) and generates
stunning interactive 2D (`inditrade_graph.html`) and 3D (`inditrade_graph_3d.html`) HTML visualizations.
"""

import sqlite3
import json
import os

DB_PATH = ".agents/graph_memory.sqlite"

def load_graph_data():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Graph database not found at {DB_PATH}")
        
    db = sqlite3.connect(DB_PATH)
    nodes_raw = db.execute("SELECT id, label, properties, trust_score FROM Nodes WHERE is_deleted=0 OR is_deleted IS NULL").fetchall()
    edges_raw = db.execute("SELECT source_id, target_id, relation_type FROM Edges").fetchall()
    
    nodes = []
    for row in nodes_raw:
        props = {}
        try:
            props = json.loads(row[2]) if row[2] else {}
        except:
            pass
        nodes.append({
            "id": row[0],
            "label": row[1],
            "title": props.get("description", props.get("purpose", props.get("path", row[0]))),
            "trust_score": row[3] if row[3] is not None else 1.0
        })
        
    edges = []
    for row in edges_raw:
        edges.append({
            "source": row[0],
            "target": row[1],
            "label": row[2]
        })
        
    db.close()
    return nodes, edges

def get_color(label):
    colors = {
        "Project": "#ff4b4b",
        "MOC_Hub": "#00f0ff",
        "File": "#00e676",
        "Component": "#ffb300",
        "External_Dependency": "#ba68c8",
        "Data_Store": "#ff007f",
        "Concept": "#29b6f6"
    }
    return colors.get(label, "#9e9e9e")

def export_2d_html(nodes, edges, output_path="inditrade_graph.html"):
    vis_nodes = []
    for n in nodes:
        vis_nodes.append({
            "id": n["id"],
            "label": n["id"].replace("File_", "").replace("Func_", "").replace("MOC_", "")[:25],
            "title": f"<b>[{n['label']}] {n['id']}</b><br/>{n['title']}<br/>Trust Score: {n['trust_score']}",
            "color": get_color(n["label"]),
            "shape": "dot" if n["label"] in ["Component", "External_Dependency"] else "box",
            "size": 25 if n["label"] == "Project" else (18 if "MOC" in n["label"] else 12)
        })
        
    vis_edges = []
    for e in edges:
        vis_edges.append({
            "from": e["source"],
            "to": e["target"],
            "label": e["label"],
            "arrows": "to",
            "color": {"color": "rgba(255,255,255,0.2)"}
        })
        
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>IndiTrade AI - Epistemic Graph Memory (2D)</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{ background-color: #11131f; margin: 0; padding: 0; overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
        #mynetwork {{ width: 100vw; height: 100vh; border: none; }}
        #legend {{ position: absolute; top: 15px; left: 15px; color: #e0e0e0; background: rgba(17, 19, 31, 0.85); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 4px 15px rgba(0,0,0,0.5); z-index: 100; }}
        #legend h3 {{ margin-top: 0; color: #00f0ff; font-size: 16px; }}
        .item {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 13px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; display: inline-block; }}
        #stats {{ position: absolute; bottom: 15px; right: 15px; color: #a0a0a0; background: rgba(17, 19, 31, 0.85); padding: 10px 15px; border-radius: 6px; font-size: 12px; border: 1px solid rgba(255,255,255,0.08); }}
    </style>
</head>
<body>
<div id="legend">
    <h3>IndiTrade AI Graph Node Types</h3>
    <div class="item"><span class="dot" style="background:#ff4b4b;"></span>Project Core</div>
    <div class="item"><span class="dot" style="background:#00f0ff;"></span>MOC Hub (Phase Module)</div>
    <div class="item"><span class="dot" style="background:#00e676;"></span>File / Script</div>
    <div class="item"><span class="dot" style="background:#ffb300;"></span>Function / Component</div>
    <div class="item"><span class="dot" style="background:#ba68c8;"></span>External Dependency</div>
    <div class="item"><span class="dot" style="background:#ff007f;"></span>Dataset / Storage</div>
</div>
<div id="stats">
    <b>Nodes:</b> {len(nodes)} | <b>Edges:</b> {len(edges)} | <b>Engine:</b> SQLite Epistemic Memory
</div>
<div id="mynetwork"></div>
<script type="text/javascript">
    var nodes = new vis.DataSet({json.dumps(vis_nodes)});
    var edges = new vis.DataSet({json.dumps(vis_edges)});
    var container = document.getElementById('mynetwork');
    var data = {{ nodes: nodes, edges: edges }};
    var options = {{
        nodes: {{ font: {{ color: '#ffffff', size: 12, face: 'Segoe UI' }} }},
        edges: {{ font: {{ color: '#888888', size: 10, align: 'middle' }}, smooth: {{ type: 'continuous' }} }},
        physics: {{
            forceAtlas2Based: {{ gravitationalConstant: -60, centralGravity: 0.01, springLength: 120, springConstant: 0.08 }},
            maxVelocity: 50,
            solver: 'forceAtlas2Based',
            timestep: 0.35,
            stabilization: {{ iterations: 150 }}
        }},
        interaction: {{ hover: true, tooltipDelay: 100 }}
    }};
    var network = new vis.Network(container, data, options);
</script>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Exported 2D Graph HTML -> {output_path}")

def export_3d_html(nodes, edges, output_path="inditrade_graph_3d.html"):
    graph_data = {
        "nodes": [{"id": n["id"], "name": n["id"], "group": n["label"], "val": 15 if n["label"]=="Project" else (10 if "MOC" in n["label"] else 5), "color": get_color(n["label"]), "desc": n["title"]} for n in nodes],
        "links": [{"source": e["source"], "target": e["target"], "label": e["label"]} for e in edges]
    }
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>IndiTrade AI - Epistemic Graph Memory (3D Space)</title>
    <script src="https://unpkg.com/3d-force-graph@1.73.2/dist/3d-force-graph.min.js"></script>
    <style>
        body {{ margin: 0; overflow: hidden; background-color: #0b0d17; font-family: 'Segoe UI', sans-serif; }}
        #legend {{ position: absolute; top: 20px; left: 20px; background: rgba(11, 13, 23, 0.85); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); color: #fff; z-index: 10; }}
        #legend h3 {{ margin-top: 0; color: #00f0ff; font-size: 16px; }}
        .item {{ display: flex; align-items: center; margin-bottom: 8px; font-size: 13px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; display: inline-block; }}
        #stats {{ position: absolute; bottom: 20px; right: 20px; color: #a0a0a0; background: rgba(11, 13, 23, 0.85); padding: 10px 15px; border-radius: 6px; font-size: 12px; border: 1px solid rgba(255,255,255,0.08); z-index: 10; }}
    </style>
</head>
<body>
<div id="legend">
    <h3>IndiTrade AI Graph Node Types (3D)</h3>
    <div class="item"><span class="dot" style="background:#ff4b4b;"></span>Project Core</div>
    <div class="item"><span class="dot" style="background:#00f0ff;"></span>MOC Hub (Phase Module)</div>
    <div class="item"><span class="dot" style="background:#00e676;"></span>File / Script</div>
    <div class="item"><span class="dot" style="background:#ffb300;"></span>Function / Component</div>
    <div class="item"><span class="dot" style="background:#ba68c8;"></span>External Dependency</div>
    <div class="item"><span class="dot" style="background:#ff007f;"></span>Dataset / Storage</div>
</div>
<div id="stats">
    <b>Nodes:</b> {len(nodes)} | <b>Edges:</b> {len(edges)} | <b>Interaction:</b> Drag to rotate / Scroll to zoom
</div>
<div id="3d-graph"></div>
<script>
    const gData = {json.dumps(graph_data)};
    const Graph = ForceGraph3D()
        (document.getElementById('3d-graph'))
        .graphData(gData)
        .nodeColor(d => d.color)
        .nodeLabel(d => `<b>[${{d.group}}] ${{d.name}}</b><br/>${{d.desc}}`)
        .nodeVal(d => d.val)
        .linkWidth(1.2)
        .linkOpacity(0.35)
        .linkDirectionalArrowLength(3.5)
        .linkDirectionalArrowRelPos(1)
        .backgroundColor('#0b0d17');
        
    Graph.d3Force('charge').strength(-120);
</script>
</body>
</html>"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[OK] Exported 3D Graph HTML -> {output_path}")

if __name__ == "__main__":
    print("=== Exporting IndiTrade AI Epistemic Graph Memory to HTML ===")
    n, e = load_graph_data()
    print(f"Loaded {len(n)} nodes and {len(e)} edges from {DB_PATH}")
    export_2d_html(n, e)
    export_3d_html(n, e)
    print("[SUCCESS] Graph memory HTML export complete!")
