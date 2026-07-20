from fastapi import APIRouter
import logging

logger = logging.getLogger("api.network")
router = APIRouter()

network_model = None

def load_model():
    global network_model
    if network_model is None:
        try:
            import joblib
            logger.info("Lazy-loading Node2Vec network embeddings...")
            network_model = joblib.load("models/node2vec_trade_graph.pkl")
            logger.info("Network model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Network model: {e}")
            network_model = "FAILED"

@router.get("/{partner_code}")
async def get_network(partner_code: str):
    load_model()
    if network_model == "FAILED":
        return {"error": "Network model is unavailable."}
        
    try:
        # network_model is expected to be a gensim Word2Vec KeyedVectors object
        if partner_code in network_model:
            vector = network_model[partner_code].tolist()
            # Also get top 3 similar partners
            similar = network_model.most_similar(partner_code, topn=3)
            return {
                "partner": partner_code,
                "embedding": vector[:5], # truncate for demo response size
                "similar_partners": similar,
                "status": "success"
            }
        else:
            return {"error": f"Partner code {partner_code} not found in graph."}
    except Exception as e:
        logger.error(f"Network graph error: {e}")
        return {"error": str(e)}
