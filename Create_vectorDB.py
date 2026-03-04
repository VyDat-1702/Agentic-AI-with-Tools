import pandas as pd
from pathlib import Path
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
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
    EMBEDDING_MODEL = "BAAI/bge-m3"
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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def create_vector_db(df, collection_name, batch_size):

    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    logger.info("Connected to Qdrant Cloud")

    logger.info(f"Loading model {config.EMBEDDING_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(config.EMBEDDING_MODEL)

    model.to(config.DEVICE)
    model.eval()

    if config.DEVICE == "cuda":
        model = model.half()

    embedding_dim = model.config.hidden_size
    logger.info(f"Embedding dimension: {embedding_dim}")

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        logger.warning(f"Collection already exists or error: {e}")

    texts = df['combined'].tolist()
    embeddings = []

    logger.info(f"Encoding {len(texts)} documents...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        batch_texts = ["passage: " + t for t in batch_texts]

        encoded_input = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(config.DEVICE)

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        embeddings.extend(sentence_embeddings.cpu().numpy())

    points = []
    for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
        point = PointStruct(
            id=str(uuid5(NAMESPACE_DNS, text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                "question": df.iloc[idx].get('Question', ""),
                "answer": df.iloc[idx].get('Answer', ""),
                "qtype": df.iloc[idx].get('qtype', "")
            }
        )
        points.append(point)

    logger.info(f"Uploading {len(points)} points to Qdrant...")

    client.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=64,
        parallel=4,
        wait=True,
    )

    logger.info(f"Successfully added {len(points)} documents to collection '{collection_name}'")

def main():
    parser = argparse.ArgumentParser(description='Medical Q&A KB with Qdrant + BGE-M3')
    parser.add_argument('--dir', type=str, default='Data',
                        help='Directory containing CSV files')
    parser.add_argument('--collection', type=str, default='medical_qa_kb',
                        help='Collection name in Qdrant')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
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