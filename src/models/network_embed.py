"""
IndiTrade AI - Module C: Bilateral Trade Network Graph & Node2Vec Embeddings
Constructs a weighted bipartite trade graph connecting trade partners (partnerCode) to commodities (cmdCode)
weighted by standardized transaction volume (primaryValue).
Runs Node2Vec random walk embeddings (64 dimensions) to capture structural trade equivalence across global supply chains.
Export: graph_nodes.parquet, graph_edges.parquet, node2vec_trade_embeddings.parquet, and .pkl model.
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import networkx as nx
from node2vec import Node2Vec

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def build_trade_graph(data_path):
    print(f"[*] Loading trade features from: {data_path}")
    df = pd.read_parquet(data_path)
    
    # Fill null descriptions to prevent pandas groupby dropna=True from dropping rows
    df['partnerDesc'] = df['partnerDesc'].fillna(df['partnerCode'].astype(str))
    df['cmdDesc'] = df['cmdDesc'].fillna('HS Code ' + df['cmdCode'].astype(str))
    
    # Aggregate bilateral trade volume across all years by partner and commodity
    print("[*] Aggregating bilateral trade volume to construct bipartite country-commodity graph...")
    edges_df = df.groupby(['partnerCode', 'partnerDesc', 'cmdCode', 'cmdDesc'], dropna=False, as_index=False).agg(
        total_value=('primaryValue', 'sum'),
        total_weight=('netWgt', 'sum'),
        trade_count=('primaryValue', 'count')
    ).copy()
    
    # Filter out zero/negative weights
    edges_df = edges_df[edges_df['total_value'] > 0].reset_index(drop=True)
    
    # Create distinct string prefixes for node IDs so partnerCode '85' does not collide with cmdCode '85'
    edges_df['source_id'] = 'P_' + edges_df['partnerCode'].astype(str)
    edges_df['target_id'] = 'C_' + edges_df['cmdCode'].astype(str)
    
    # Log-transform weight so massive multi-billion flows don't completely trap random walks
    edges_df['weight'] = np.log1p(edges_df['total_value'])
    
    # Build NetworkX graph
    G = nx.Graph()
    
    partner_nodes = []
    commodity_nodes = []
    
    # Add nodes with metadata
    unique_partners = edges_df[['source_id', 'partnerCode', 'partnerDesc']].drop_duplicates()
    for _, row in unique_partners.iterrows():
        G.add_node(row['source_id'], node_type='partner', desc=row['partnerDesc'], raw_id=row['partnerCode'])
        partner_nodes.append({
            'node_id': row['source_id'],
            'raw_id': str(row['partnerCode']),
            'node_type': 'partner',
            'node_desc': row['partnerDesc']
        })
        
    unique_commodities = edges_df[['target_id', 'cmdCode', 'cmdDesc']].drop_duplicates()
    for _, row in unique_commodities.iterrows():
        G.add_node(row['target_id'], node_type='commodity', desc=row['cmdDesc'], raw_id=row['cmdCode'])
        commodity_nodes.append({
            'node_id': row['target_id'],
            'raw_id': str(row['cmdCode']),
            'node_type': 'commodity',
            'node_desc': str(row['cmdDesc'])[:60]
        })
        
    # Add weighted edges
    for _, row in edges_df.iterrows():
        G.add_edge(row['source_id'], row['target_id'], weight=row['weight'], raw_value=row['total_value'])
        
    nodes_df = pd.DataFrame(partner_nodes + commodity_nodes)
    print(f"[OK] Graph constructed successfully: {G.number_of_nodes()} unique nodes | {G.number_of_edges()} bilateral trade edges.")
    return G, nodes_df, edges_df

def main():
    parser = argparse.ArgumentParser(description="IndiTrade AI - Node2Vec Bilateral Trade Network Embedder")
    parser.add_argument("--data-path", type=str, default="data/processed/trade_features.parquet", help="Path to trade_features.parquet")
    parser.add_argument("--dimensions", type=int, default=64, help="Embedding vector dimensions (default: 64)")
    parser.add_argument("--walk-length", type=int, default=30, help="Random walk length (default: 30)")
    parser.add_argument("--num-walks", type=int, default=100, help="Number of walks per node (default: 100)")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save exported parquet files")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save exported .pkl node2vec model")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    print("=== [PHASE 3: MODULE C] NODE2VEC BILATERAL TRADE NETWORK GRAPH EMBEDDINGS ===")
    
    G, nodes_df, edges_df = build_trade_graph(args.data_path)
    
    print(f"\n[*] Running Node2Vec (dimensions={args.dimensions}, walk_length={args.walk_length}, num_walks={args.num_walks}, p=1.0, q=0.5)...")
    node2vec = Node2Vec(
        G,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=1.0,  # Return hyperparameter
        q=0.5,  # In-out hyperparameter (biased toward structural equivalence / outward exploration)
        workers=4,
        quiet=False
    )
    
    print("[*] Fitting Word2Vec skip-gram model on trade network random walks...")
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    # Extract embeddings into structured DataFrame
    print("[*] Extracting 64-dimensional structural trade embeddings...")
    embedding_rows = []
    for _, row in nodes_df.iterrows():
        n_id = row['node_id']
        if n_id in model.wv:
            vec = model.wv[n_id].tolist()
            embedding_rows.append({
                'node_id': n_id,
                'raw_id': row['raw_id'],
                'node_type': row['node_type'],
                'node_desc': row['node_desc'],
                'embedding_vector': vec
            })
            
    embeddings_df = pd.DataFrame(embedding_rows)
    print(f"[OK] Extracted embeddings for {len(embeddings_df)} graph nodes.")
    
    # Export Parquet datasets
    nodes_parquet = os.path.join(args.output_dir, "graph_nodes.parquet")
    edges_parquet = os.path.join(args.output_dir, "graph_edges.parquet")
    embed_parquet = os.path.join(args.output_dir, "node2vec_trade_embeddings.parquet")
    
    nodes_df.to_parquet(nodes_parquet, index=False)
    edges_df.to_parquet(edges_parquet, index=False)
    embeddings_df.to_parquet(embed_parquet, index=False)
    
    print("\n=== PARQUET EXPORT SUMMARY ===")
    print(f"  ✅ {nodes_parquet} ({len(nodes_df)} nodes)")
    print(f"  ✅ {edges_parquet} ({len(edges_df)} weighted edges)")
    print(f"  ✅ {embed_parquet} ({len(embeddings_df)} 64-dim embeddings)")
    
    # Export model checkpoint
    model_pkl = os.path.join(args.model_dir, "node2vec_trade_graph.pkl")
    joblib.dump({"wv": model.wv, "graph": G, "dimensions": args.dimensions}, model_pkl)
    print(f"  ✅ {model_pkl} (trained Word2Vec/Node2Vec checkpoint)")
    
    print("\n=== SAMPLE STRUCTURAL SIMILARITY LOOKUP (TOP 3 SIMILAR TO 'USA') ===")
    # Find USA partner node if present
    usa_nodes = embeddings_df[(embeddings_df['node_type'] == 'partner') & (embeddings_df['node_desc'].str.contains('USA|United States', case=False, na=False))]
    if not usa_nodes.empty:
        usa_id = usa_nodes.iloc[0]['node_id']
        similar_nodes = model.wv.most_similar(usa_id, topn=3)
        print(f"Most structurally equivalent trade nodes to [{usa_nodes.iloc[0]['node_desc']} ({usa_id})]:")
        for sim_id, sim_score in similar_nodes:
            sim_info = embeddings_df[embeddings_df['node_id'] == sim_id]
            if not sim_info.empty:
                print(f"  * {sim_info.iloc[0]['node_desc']} ({sim_info.iloc[0]['node_type']}) | Cosine Sim: {sim_score:.4f}")
            else:
                print(f"  * {sim_id} | Cosine Sim: {sim_score:.4f}")
                
    print("\n🎯 Module C (Node2Vec Trade Network Embeddings) complete!")

if __name__ == "__main__":
    main()
