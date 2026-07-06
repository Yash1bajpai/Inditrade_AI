#!/bin/bash
# Export Lightning AI / Camber main environment to PATH
export PATH=/opt/jupyter/envs/main/bin:$PATH
# ==========================================
# Camber / Lightning AI Job — Network Graph
# ==========================================
# Computes PageRank, Betweenness, and Node2Vec embeddings for TradeNet graph.
# Target: 4 CPU cores, 16GB RAM engine (~4 CPU hours)

echo "[+] Starting NetworkX & Node2Vec Graph Embedding Job..."
pip install networkx node2vec pandas pyarrow numpy --quiet

python -c "
from src.models.network_embed import TradeNetworkEmbedder
embedder = TradeNetworkEmbedder()
e_out, n_out, emb_out = embedder.build_graph_and_embed()
print(f'\n[✓] Graph Job Complete!\n  - Edges: {e_out}\n  - Nodes: {n_out}\n  - Embeddings: {emb_out}')
"
