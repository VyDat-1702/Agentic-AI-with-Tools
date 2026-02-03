import pandas as pd
import json
from pathlib import Path
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import torch
import os
from dotenv import load_dotenv
from uuid import uuid5, NAMESPACE_DNS
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()

logger.info(f"Using device: {config.DEVICE}")
logger.info(f"Qdrant URL: {config.QDRANT_URL}")


def load_csvs_from_dir(directory):
    csv_files = list(Path(directory).glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    
    logger.info(f"Found {len(csv_files)} CSV file(s)")
    
    dfs = []
    for file in csv_files:
        logger.info(f"  ├─ Loading {file.name}")
        dfs.append(pd.read_csv(file))
    
    return pd.concat(dfs, ignore_index=True)


def prepare_documents(df):
    df = df.fillna("")

    df['combined'] = df.apply(
        lambda row: ". ".join(
            f"{col}: {row[col]}" for col in df.columns
        ),
        axis=1
    )
    return df


def create_vector_db(df, collection_name, batch_size):

    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    logger.info("Connected to Qdrant Cloud")
    
    encoder = SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)
    logger.info(f"Encoder running on: {encoder.device}")

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=encoder.get_sentence_embedding_dimension(),  
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Collection already exists or error: {e}")

    points = []
    texts = df['combined'].tolist()
    
    logger.info(f"Encoding {len(texts)} documents...")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True,
        device=config.DEVICE
    )
    
    for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
        point = PointStruct(
            id=str(uuid5(NAMESPACE_DNS, text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                "question": df.iloc[idx]['Question'],
                "answer": df.iloc[idx]['Answer'],
                "qtype": df.iloc[idx]['qtype']
            }
        )
        points.append(point)
    
    logger.info(f"Uploading {len(points)} points to Qdrant Cloud...")
    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=4,
        wait=True,
    )
    
    logger.info(f"Successfully added {len(points)} documents to collection '{collection_name}'")
    


def main():
    parser = argparse.ArgumentParser(description='Medical Q&A KB with Qdrant Cloud')
    parser.add_argument('--dir', type=str, default='Data',
                        help='Directory containing CSV files')
    parser.add_argument('--collection', type=str, default='medical_qa_kb',
                        help='Collection name in Qdrant')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for encoding (GPU: 64-128, CPU: 16-32)')
    parser.add_argument('--test', action='store_true',
                        help='Run test search after uploading')
    args = parser.parse_args()

    if not config.QDRANT_URL or not config.QDRANT_API_KEY:
        raise ValueError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")
    
    df = load_csvs_from_dir(args.dir)
    
    df = prepare_documents(df)
    

    create_vector_db(
        df, 
        args.collection,
        batch_size=args.batch_size
    )
if __name__ == '__main__':
    main()