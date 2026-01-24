import pandas as pd
import json
from pathlib import Path
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import torch

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {DEVICE}")


def load_csvs_from_dir(directory):
    csv_files = list(Path(directory).glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"Không tìm thấy CSV nào trong {directory}")
    
    print(f" Found {len(csv_files)} file(s) CSV")
    
    dfs = []
    for file in csv_files:
        print(f"   ├─ {file.name}")
        dfs.append(pd.read_csv(file))
    
    return pd.concat(dfs, ignore_index=True)


def prepare_documents(df):
    required_cols = ['Question', 'Answer', 'qtype']
    
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV phải có các cột: {required_cols}")
    
    df['combined_text'] = (
        "Question: " + df['Question'].astype(str) + ". " +
        "Answer: " + df['Answer'].astype(str) + ". " +
        "Type: " + df['qtype'].astype(str) + "."
    )
    
    return df


def create_vector_db(df, collection_name, qdrant_path=":memory:", device=DEVICE, batch_size=32):
    
    client = QdrantClient(path=qdrant_path)
    encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    
    print(f" Encoder running on: {encoder.device}")

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )
        print(f" Created collection: {collection_name}")
    except Exception as e:
        print(f"  Collection already exists: {e}")

    points = []
    texts = df['combined_text'].tolist()
    
    print(f" Encoding {len(texts)} documents...")
    embeddings = encoder.encode(
        texts,
        batch_size=batch_size, 
        show_progress_bar=True,
        convert_to_numpy=True,
        device=device
    )
    
    for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
        point = PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "text": text,
                "question": df.iloc[idx]['Question'],
                "answer": df.iloc[idx]['Answer'],
                "qtype": df.iloc[idx]['qtype']
            }
        )
        points.append(point)
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    print(f" Add {len(points)} documents to collection '{collection_name}'")
    
    return client, encoder


def search_kb(client, encoder, collection_name, query, n_results=5):
    query_vector = encoder.encode(query, convert_to_numpy=True).tolist()
    
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=n_results
    ).points 
    
    formatted_results = {
        "ids": [[r.id for r in results]],
        "scores": [[r.score for r in results]],
        "documents": [[r.payload.get("text", "") for r in results]],
        "metadatas": [[{
            "question": r.payload.get("question", ""),
            "answer": r.payload.get("answer", ""),
            "qtype": r.payload.get("qtype", "")
        } for r in results]]
    }
    
    return formatted_results, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Medical Q&A KB with Qdrant')
    parser.add_argument('--dir', type=str, default='Data',
                        help='Directory containing CSV files')
    parser.add_argument('--colna', type=str, default='medical_qa_kb',
                        help='Collection name')
    parser.add_argument('--qdrant_path', type=str, default='QdrantDB',
                        help='Qdrant storage path')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device for embedding model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding (GPU: 64-128, CPU: 16-32)')
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"\n Selected device: {device.upper()}")
    
    print("\n Loading data...")
    df = load_csvs_from_dir(args.dir)
    print(f" Total records: {len(df)}\n")
    
    df = prepare_documents(df)
    
    print("\n Building vector database...")
    client, encoder = create_vector_db(
        df, 
        args.colna, 
        args.qdrant_path,
        device=device,
        batch_size=args.batch_size
    )
    
    print("\n Testing search...")
    query = "What are the most common symptoms of diabetes?"
    formatted_results, raw_results = search_kb(client, encoder, args.colna, query)
    
    print(f"\n Query: {query}")
    print("\n Top 3 Results:")
    print()
    
    for i, (meta, score) in enumerate(zip(formatted_results['metadatas'][0][:3], 
                                           formatted_results['scores'][0][:3]), 1):
        print(f"\n{i}. [Score: {score:.4f}]")
        print(f"   Q: {meta['question'][:100]}...")
        print(f"   A: {meta['answer'][:150]}...")
        print(f"   Type: {meta['qtype']}")
    
    print()
    print(f" Done! Collection '{args.colna}' ready on {device.upper()}")
    print(f" Saved to: {args.qdrant_path}")