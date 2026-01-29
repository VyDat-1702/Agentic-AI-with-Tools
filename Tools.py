import os
import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict
import logging
import requests

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config():
    QDRANT_URL: str = os.getenv('QDRANT_URL')
    QDRANT_API_KEY: str = os.getenv('QDRANT_API_KEY')
    COLLECTIONS: str = os.getenv('COLLECTIONS')
    EMBEDDING_MODEL: str = 'all-MiniLM-L6-v2'
    
    TOP_K: int = 3
    WEB_SEARCH_NUM: int = 5

    SERPAPI_KEY: str = os.getenv('SERPAPI_KEY')
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
class QA_Retriever:
    def __init__(self, config):
        self.config = config
        self.qdrant_client = QdrantClient(
            url=config.QDRANT_URL, 
            api_key=config.QDRANT_API_KEY
            )
        for i in self.qdrant_client.get_collections():
            logger.info(f'Connect to {i} collection')
            
        self.encoder = SentenceTransformer(
            model_name_or_path=config.EMBEDDING_MODEL,
            device=config.DEVICE
        )
        
    def search(self, query: str, top_k: Optional[int] = None) -> Dict:
        try:
            query_vector = self.encoder.encode(
                sentences=query,
                convert_to_tensor=(self.config.DEVICE == 'cuda'),
                convert_to_numpy=(self.config.DEVICE =='cpu'),
                batch_size=128,
                normalize_embeddings=True,
                show_progress_bar=True,
                device=self.config.DEVICE,
            ).tolist()
            
            results = self.qdrant_client.query_points(
                collection_name='Comprehensive-Medical-QA',
                query=query_vector,
                limit=top_k if top_k is not None else self.config.TOP_K,
            ).points
            
            if not results:
                return {
                'context': 'No relevant QA found',
                'source': 'Medical QA KB',
                'result': []
                }
                
            context_parts=[]
            results_details = []
            
            for i, point in enumerate(results, 1):
                score = point.score
                text = point.payload.get('Text', 'N/A')
                
                context_parts.append(
                    f'[{i}] | Score: {score:.3f} | {text}'
                )
                results_details.append({
                    'Question': point.payload.get('Question', 'N/A'),
                    'Answer': point.payload.get('Answer', 'N/A'),
                    'qtype': point.payload.get('qtype', 'N/A'),
                    'score': score
                }
                )
            context = '\n\n'.join(context_parts)
            
            return {
                'context': context, #context for LLM, save token
                'source': 'Comprehensive-Medical-QA', #citations
                'results': results_details, #for displaydisplay UI
                'top_score': results[0].score
            }
        
        except Exception as e:
            logger.info(f'QA search ERROR: {str(e)}')
            return {
                'contextf': f'QA seech error: {str(e)}',
                'source': 'Error',
                'result': []
            }
            
            
class WebSearcher:

    def __init__(self, config: Config = Config()):
        self.config = config
        self.api_key = config.SERPAPI_KEY
        self.url = "https://serpapi.com/search"
        
        if not self.api_key:
            logger.info("Warning: SERPAPI_KEY not found in .env file")
    
    def search(self, query: str) -> Dict:   
        if not self.api_key:
            return {
                "context": " SerpAPI key not configured. Add SERPAPI_KEY to .env file.",
                "source": "Web Search Error",
                "num_results": 0
            }
        
        try:
            params = {
                "q": f"{query} medical health",
                "api_key": self.api_key,
                "num": self.config.WEB_SEARCH_NUM,
                "gl": "vn",
                "hl": "en"
            }
            
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic_results", [])[:self.config.WEB_SEARCH_NUM]:
                results.append(
                    f"Title: {item.get('title')}\n"
                    f"Source: {item.get('link')}\n"
                    f"Summary: {item.get('snippet')}"
                )
            
            context = "\n\n".join(results) if results else "No web results found"
            
            return {
                "context": context,
                "source": "Web Search (SerpAPI)",
                "num_results": len(results)
            }
            
        except Exception as e:
            logger.info(f" Web search error: {e}")
            return {
                "context": f"Web search unavailable: {str(e)}",
                "source": "Web Search Error",
                "num_results": 0
            }
            
config_arg = Config()

def get_qa_retriever(query: str) -> Dict:
    global config_arg
    return QA_Retriever(config_arg).search(query)

def get_web_search(query: str) -> Dict:
    global config_arg
    return WebSearcher(config_arg).search(query)
    
TOOLS_MAPPING_TO_FUNC = {
    'get_qa_retriever': get_qa_retriever,
    'get_web_search': get_web_search
}

AGENT_TOOLS_LIST = {
    'TOOLS': [
        {
            'name': 'get_qa_retriever',
            'description': 'Retrieve relevant Medical QA in knowledge base.',
            'args': 'query(str)'
        },
        {
            'name': 'get_web_search',
            'description': 'Search relevant information from website',
            'args': 'query (str)'
        }
    ]
}