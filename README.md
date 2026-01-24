# Medical Q&A Agent with RAG

A medical information assistant powered by Google Gemini, Qdrant vector database, and LangGraph for agentic workflows. The system uses Retrieval-Augmented Generation (RAG) to answer medical queries from a knowledge base of 16K+ Q&A pairs, with fallback to web search for current information.

## Features

- **Vector Database**: Fast semantic search over medical Q&A using Qdrant
- **Dual Information Sources**:
  - Local FAQ retrieval from vector database
  - Web search fallback via SerpAPI
- **Agentic Workflow**: LangGraph-powered decision making
- **Safety First**: Built-in disclaimers and emergency handling
- **GPU Acceleration**: CUDA support for faster embeddings

## Architecture

```
User Query
    ↓
Agent (Gemini)
    ↓
Tool Selection
    ├─→ get_medical_faq (Primary)
    └─→ web_search_medical (Fallback)
    ↓
Response Generation + Disclaimer
```

## Project Structure

```
.
├── Create_vectorDB.py          # Vector database creation script
├── main.py                      # Agent orchestration and workflow
├── Tools.py                     # FAQ retriever and web searcher
├── requirements.txt             # Python dependencies
├── Data/
│   └── Comprehensive-Medical-Q&A.csv
└── QdrantDB/                    # Vector database storage
    ├── collection/
    └── meta.json
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA (optional, for GPU acceleration)

### Setup

1. **Clone the repository**

```bash
git clonehttps://github.com/VyDat-1702/Agentic-AI-with-Tools.git
cd Agentic-AI-with-Tools/
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure environment variables**

Create a `.env` file in the project root:

```env
# Required
GEMINI_API_KEY=your_gemini_api_key_here

# Optional
SERPAPI_KEY=your_serpapi_key_here
VECTOR_DB_PATH=./QdrantDB
COLLECTION_NAME=medical_qa_kb
```

**Get API Keys:**

- Gemini API: [Google AI Studio](https://aistudio.google.com/apikey)
- SerpAPI (optional): [SerpAPI Dashboard](https://serpapi.com/manage-api-key)

4. **Prepare your data**

Place your CSV file(s) in the `Data/` directory with these columns:

- `Question`: Medical question
- `Answer`: Corresponding answer
- `qtype`: Question type/category

## Usage

### Step 1: Build Vector Database

```bash
python Create_vectorDB.py --dir Data --colna medical_qa_kb --qdrant_path QdrantDB
```

**Arguments:**

- `--dir`: Directory containing CSV files (default: `Data`)
- `--colna`: Collection name (default: `medical_qa_kb`)
- `--qdrant_path`: Database storage path (default: `QdrantDB`)
- `--device`: Device for embeddings - `auto`, `cuda`, or `cpu` (default: `auto`)
- `--batch_size`: Encoding batch size (default: 32, use 64-128 for GPU)

**Example with GPU:**

```bash
python Create_vectorDB.py --device cuda --batch_size 128
```

### Step 2: Run the Agent

```bash
python main.py
```

The script will run test queries automatically. You can modify the queries in `main.py`:

```python
queries = [
    "What are symptoms of diabetes?",
    "How to prevent heart disease?",
    "Latest COVID-19 treatment options?",
]
```

### Custom Queries

Modify the `main()` function in `main.py`:

```python
def main():
    initialize_tools()
    graph = build_graph()

    # Your custom query
    run_query("What are the side effects of aspirin?", graph)
```

## Components

### 1. Create_vectorDB.py

**Purpose**: Creates and indexes medical Q&A into Qdrant vector database

**Key Functions:**

- `load_csvs_from_dir()`: Load CSV files from directory
- `prepare_documents()`: Combine Q&A into searchable text
- `create_vector_db()`: Encode and store in Qdrant
- `search_kb()`: Test semantic search

**Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions)

### 2. Tools.py

**Purpose**: Provides search tools for the agent

**Classes:**

- `FAQRetriever`: Searches local vector database
- `WebSearcher`: Searches web via SerpAPI

**Functions:**

- `get_medical_faq(query)`: Primary search tool
- `web_search_medical(query)`: Fallback web search

### 3. main.py

**Purpose**: Agent orchestration using LangGraph

**Workflow:**

1. **Agent Node**: Analyzes query, decides on tool
2. **Tool Node**: Executes selected tool
3. **Decision**: Continue or provide final answer

**Agent Instructions:**

- Always provide safety disclaimers
- Never diagnose or prescribe
- Try FAQ search first
- Use web search if FAQ insufficient

## Configuration

### Config Class (Tools.py)

```python
@dataclass
class Config:
    VECTOR_DB_PATH: str = "./QdrantDB"
    COLLECTION_NAME: str = "medical_qa_kb"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    FAQ_TOP_K: int = 3
    WEB_SEARCH_NUM: int = 3
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
```

### Gemini Settings (main.py)

```python
generation_config=genai.types.GenerationConfig(
    temperature=0.1,        # Low for factual responses
    max_output_tokens=1024,
)
```

## Safety Features

The agent includes several safety mechanisms:

1. **Mandatory Disclaimer**: Every response includes "This is for informational purposes only. Consult a healthcare professional."

2. **No Diagnosis/Prescription**: Agent refuses to diagnose conditions or prescribe medications

3. **Emergency Handling**: For emergency symptoms, advises calling 115 or visiting ER

4. **Step Limit**: Maximum 10 reasoning steps to prevent infinite loops

## Example Output

```
💬 USER: What are symptoms of diabetes?

🤖 AGENT (Step 1)
THOUGHT: This is a common medical question that should be in the FAQ database
ACTION: get_medical_faq
ARGUMENTS: {"query": "symptoms of diabetes"}

⚙️ EXECUTING TOOL
Tool: get_medical_faq
Found 3 results

🤖 AGENT (Step 2)
THOUGHT: I have sufficient information from the FAQ database
ANSWER: Common symptoms of diabetes include:
- Increased thirst and frequent urination
- Extreme hunger
- Unexplained weight loss
- Fatigue
- Blurred vision
- Slow-healing sores
- Frequent infections

This is for informational purposes only. Consult a healthcare professional
for proper diagnosis and treatment.
```

## Troubleshooting

### "Vector DB not found" Error

```bash
# Run the vector DB creation script first
python Create_vectorDB.py
```

### "GEMINI_API_KEY not found" Error

```bash
# Add your API key to .env file
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Slow Performance

```bash
# Use GPU acceleration
python Create_vectorDB.py --device cuda --batch_size 128
```

### Web Search Not Working

- Check if `SERPAPI_KEY` is set in `.env`
- The system will still work with FAQ-only mode

## Performance

**Vector Database Creation:**

- CPU: ~2-5 minutes for 16K documents
- GPU (CUDA): ~30-60 seconds

**Query Response Time:**

- FAQ search: ~100-300ms
- With web search: ~2-5 seconds

## Limitations

- Requires internet for Gemini API and web search
- Vector DB must be rebuilt if data changes
- SerpAPI has rate limits (100 searches/month free tier)
- Responses are informational only, not medical advice

## Future Improvements

- [ ] Add conversation memory
- [ ] Support multi-turn dialogues
- [ ] Implement source citation
- [ ] Add medical image analysis
- [ ] Support multiple languages
- [ ] Fine-tune embeddings on medical domain
