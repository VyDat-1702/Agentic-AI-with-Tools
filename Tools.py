import pandas as pd
import requests
import os
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    # Qdrant settings
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./QdrantDB")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "medical_qa_kb")
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Search settings
    FAQ_TOP_K: int = 5
    WEB_SEARCH_NUM: int = 3
    
    # API keys from environment
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

class FAQRetriever:
    def __init__(self, config: Config = Config()):
        self.config = config
        
        if not Path(config.VECTOR_DB_PATH).exists():
            raise FileNotFoundError(
                f"Vector DB not found at: {config.VECTOR_DB_PATH}\n"
                f"Please run Create_vectorDB.py first!"
            )

        self.client = QdrantClient(path=config.VECTOR_DB_PATH)
        

        self.encoder = SentenceTransformer(
            config.EMBEDDING_MODEL,
            device=config.DEVICE
        )
        

        try:
            collection_info = self.client.get_collection(config.COLLECTION_NAME)
            print(f" Connected to collection '{config.COLLECTION_NAME}'")
            print(f" Documents: {collection_info.points_count}")
        except Exception as e:
            raise ValueError(
                f" Collection '{config.COLLECTION_NAME}' not found!\n"
                f"   Error: {e}"
            )
    
    def search(self, query: str, top_k: Optional[int] = None) -> Dict:
        if top_k is None:
            top_k = self.config.FAQ_TOP_K
        
        try:
            # Encode query
            query_vector = self.encoder.encode(
                query,
                convert_to_numpy=True
            ).tolist()
            
            # Search Qdrant
            results = self.client.query_points(
                collection_name=self.config.COLLECTION_NAME,
                query=query_vector,
                limit=top_k
            ).points
            
            if not results:
                return {
                    "context": "No relevant FAQ found",
                    "source": "Medical FAQ KB",
                    "results": []
                }
            
            # Format context
            context_parts = []
            result_details = []
            
            for i, result in enumerate(results, 1):
                question = result.payload.get('question', 'N/A')
                answer = result.payload.get('answer', 'N/A')
                qtype = result.payload.get('qtype', 'general')
                score = result.score
                
                context_parts.append(
                    f"[{i}] Q: {question}\n    A: {answer}\n    Type: {qtype} | Score: {score:.3f}"
                )
                
                result_details.append({
                    "question": question,
                    "answer": answer,
                    "type": qtype,
                    "score": score
                })
            
            context = "\n\n".join(context_parts)
            
            return {
                "context": context,
                "source": "Medical FAQ KB (Qdrant)",
                "results": result_details,
                "top_score": results[0].score
            }
            
        except Exception as e:
            print(f" FAQ search error: {e}")
            return {
                "context": f"FAQ search error: {str(e)}",
                "source": "Error",
                "results": []
            }

class WebSearcher:

    def __init__(self, config: Config = Config()):
        self.config = config
        self.api_key = config.SERPAPI_KEY
        self.url = "https://serpapi.com/search"
        
        if not self.api_key:
            print("  Warning: SERPAPI_KEY not found in .env file")
    
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
            print(f" Web search error: {e}")
            return {
                "context": f"Web search unavailable: {str(e)}",
                "source": "Web Search Error",
                "num_results": 0
            }


_config = Config()
faq_retriever: Optional[FAQRetriever] = None
web_searcher = WebSearcher(_config)


def initialize_tools():
    """Initialize global tools"""
    global faq_retriever, web_searcher, _config
    
    print(" Initializing tools...")
    
    try:
        faq_retriever = FAQRetriever(_config)
    except Exception as e:
        print(f" FAQ Retriever failed: {e}")
        faq_retriever = None
    
    web_searcher = WebSearcher(_config)
    print(" Web Searcher initialized")


def get_medical_faq(query: str) -> Dict:
    print(f"TOOL CALL: FAQ SEARCH")
    print(f"Query: {query}")
    
    if faq_retriever is None:
        return {
            "context": " FAQ not initialized",
            "source": "Error"
        }
    
    result = faq_retriever.search(query)
    print(f"Found {len(result.get('results', []))} results")
    
    return result


def web_search_medical(query: str) -> Dict:

    print(f"TOOL CALL: WEB SEARCH")
    print(f"Query: {query}")
    
    result = web_searcher.search(query)
    print(f"Found {result.get('num_results', 0)} results")
    
    return result

TOOLS_MAPPING_2_FUNCTIONS = {
    "get_medical_faq": get_medical_faq,
    "web_search_medical": web_search_medical
}

TOOLS_DESCRIPTION = """Available Medical Tools:

1. get_medical_faq(query: str) -> dict
   Description: Search 16K+ medical Q&A in vector database
   Use: ALWAYS try this FIRST for any medical question
   Arguments: query (str) - medical question or symptom
   Returns: FAQ answers with relevance scores

2. web_search_medical(query: str) -> dict
   Description: Search web for current medical information
   Use: When FAQ insufficient or need recent updates
   Arguments: query (str) - medical topic
   Returns: Web search results

Strategy: FAQ first → Web search if needed
"""


if __name__ == "__main__":
    print("TESTING MEDICAL TOOLS")
    
    initialize_tools()
    
    print()
    print("TEST 1: FAQ Search")
    
    result = get_medical_faq("What are symptoms of diabetes?")
    print(f"\nContext preview:\n{result['context'][:300]}...")
    
    print()
    print("TEST 2: Web Search")
    
    result = web_search_medical("Latest COVID treatment 2025")
    print(f"\nResults: {result.get('num_results', 0)}")