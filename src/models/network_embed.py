"""
Graph Analytics & Node2Vec Embedding Module for TradeNet Graph (Module C).

Constructs India-Partner trade network using NetworkX.
Computes PageRank centrality, Betweenness centrality, and Node2Vec embeddings (64-dim).
Saves graph edges/nodes to Parquet and embeddings to Numpy (.npy) for fast D3.js API response.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

try:
    import networkx as nx
    from node2vec import Node2Vec
    HAS_GRAPH = True
except ImportError:
    HAS_GRAPH = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("NetworkEmbedder")


class TradeNetworkEmbedder:
    """Precomputes graph topology metrics and Node2Vec embeddings for TradeNet D3.js visualization."""

    def __init__(self, 
                 features_path: str = "data/processed/trade_features.parquet",
                 processed_dir: str = "data/processed",
                 models_dir: str = "models"):
        self.features_path = Path(features_path)
        self.processed_dir = Path(processed_dir)
        self.models_dir = Path(models_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def build_graph_and_embed(self) -> Tuple[str, str, str]:
        """
        Constructs trade network graph, calculates PageRank & Betweenness,
        runs Node2Vec random walks, and exports static graph files for API.
        """
        if not self.features_path.exists():
            logger.error(f"Features file missing at {self.features_path}. Run trade_features.py first.")
            raise FileNotFoundError(f"Missing {self.features_path}")
            
        df = pd.read_parquet(self.features_path)
        
        # Aggregate total trade volume across all years/commodities per partner
        edges_df = df.groupby(["reporter_name", "partner_name"]).agg({
            "trade_value_usd": "sum",
            "growth_rate": "mean",
            "volatility": "mean"
        }).reset_index()
        
        edges_df.rename(columns={"reporter_name": "source", "partner_name": "target"}, inplace=True)
        
        logger.info(f"Constructed {len(edges_df)} trade network edges between India and global partners.")
        
        # Save graph edges Parquet for fast backend loading
        edges_out = self.processed_dir / "graph_edges.parquet"
        edges_df.to_parquet(edges_out, index=False, engine="pyarrow")
        
        if not HAS_GRAPH:
            logger.warning("networkx or node2vec missing. Generating mock embeddings and node metrics.")
            nodes_df = pd.DataFrame({
                "node": edges_df["target"].unique(),
                "pagerank": np.random.uniform(0.01, 0.15, len(edges_df["target"].unique())),
                "betweenness": np.random.uniform(0.0, 0.1, len(edges_df["target"].unique()))
            })
            nodes_out = self.processed_dir / "graph_nodes.parquet"
            nodes_df.to_parquet(nodes_out, index=False, engine="pyarrow")
            
            mock_embed = {node: np.random.normal(0, 1, 64) for node in nodes_df["node"]}
            mock_embed["India"] = np.random.normal(0, 1, 64)
            embed_out = self.models_dir / "node2vec_embeddings.npy"
            np.save(embed_out, mock_embed)
            return str(edges_out), str(nodes_out), str(embed_out)

        # Build NetworkX graph
        G = nx.from_pandas_edgelist(
            edges_df, "source", "target", 
            edge_attr=["trade_value_usd", "growth_rate", "volatility"],
            create_using=nx.Graph()
        )
        
        logger.info("Calculating PageRank and Betweenness Centrality...")
        pr = nx.pagerank(G, weight="trade_value_usd")
        bc = nx.betweenness_centrality(G, weight="trade_value_usd")
        
        nodes_data = []
        for node in G.nodes():
            nodes_data.append({
                "node": node,
                "pagerank": pr.get(node, 0.0),
                "betweenness": bc.get(node, 0.0),
                "degree": G.degree[node]
            })
        nodes_df = pd.DataFrame(nodes_data)
        nodes_out = self.processed_dir / "graph_nodes.parquet"
        nodes_df.to_parquet(nodes_out, index=False, engine="pyarrow")
        logger.info(f"Saved graph node metrics to {nodes_out}")
        
        logger.info("Running Node2Vec random walks (64 dimensions)...")
        node2vec = Node2Vec(G, dimensions=64, walk_length=20, num_walks=100, workers=2, quiet=True)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
        embeddings = {node: model.wv[node] for node in G.nodes()}
        embed_out = self.models_dir / "node2vec_embeddings.npy"
        np.save(embed_out, embeddings)
        logger.info(f"SUCCESS: Saved 64-dim Node2Vec embeddings for {len(embeddings)} nodes to {embed_out}")
        
        return str(edges_out), str(nodes_out), str(embed_out)


if __name__ == "__main__":
    embedder = TradeNetworkEmbedder()
    e_out, n_out, emb_out = embedder.build_graph_and_embed()
    print(f"\n[+] Network Graph & Node2Vec complete. Edges: {e_out}, Nodes: {n_out}, Embeddings: {emb_out}")
