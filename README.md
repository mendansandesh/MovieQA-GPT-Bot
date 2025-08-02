# 🎬 MovieQA-GPT-Bot 🎤

> AI-powered Question Answering Bot over Movie Transcripts using RAG (Retrieval-Augmented Generation)

MovieQA-GPT-Bot is a production-ready, Dockerized AI system that lets users ask natural language questions about movie plots or scenes. It uses vector databases and Large Language Models (LLMs) to answer contextually using actual **movie transcripts**.

---

## 🔍 Key Features

- 🎞️ Ingests and indexes full movie transcripts
- 🧠 Uses **RAG (Retrieval-Augmented Generation)** pipeline with OpenAI/GPT models
- 🔎 Semantic search using `ChromaDB` + `FAISS`
- 🐳 Fully Dockerized with `Dockerfile` and `docker-compose.yaml`
- 🧪 Modular design — easy to plug in new LLMs or datasets

---
## 🧰 Tech Stack

| Component              | Technology / Library                   |
|------------------------|----------------------------------------|
| Language & Runtime     | Python 3.10+                           |
| UI                     | Streamlit *(or React + Tailwind)*     |
| Orchestrator / Agent   | LangChain *(Agents & Chains)*         |
| Transcript Loader      | `youtube_transcript_api`              |
| Chunking & Embedding   | HuggingFace `all-MiniLM-L6-v2`        |
| Vector DB              | Chroma DB                             |

---
## 📁 Project Structure

```
movieqa_bot/
├── app.py # Entry point for the bot
├── Dockerfile # Docker setup
├── docker-compose.yaml # Multi-container orchestration
├── requirements.txt # Python dependencies
├── chroma_db/ # Chroma vector DB files
├── transcript/ # Raw movie transcript files
├── vectorstore/ # Persisted FAISS index
├── rag/ # RAG pipeline implementation
├── docker/ # Docker-specific configs/scripts
└── README.md # This file
```

---

## ⚙️ Setup & Installation

### 🧠 Prerequisites
- Python 3.8+
- Docker & Docker Compose

### 🔧 Local Setup (Without Docker)

```bash
git clone https://github.com/<your-username>/MovieQA-GPT-Bot.git
cd MovieQA-GPT-Bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run app
python app.py
```

### 🐳 Docker-Based Setup (Recommended)
#### Build and run using Docker Compose
```
1. docker compose build
	ONLY Once, at project start or if changes in requirements.txt or Dockerfile
2.docker compose run --rm app python app.py <YOUTUBE_VIDEO_ID>
	Every code/test cycle. Instant, no rebuild.
3. (Optional) export VIDEO_ID=<YOUTUBE_VIDEO_ID> and then docker compose up
	If you want your service running continuously (e.g. with a web UI).
4. docker compose down
```
---
🧠 How It Works
1. Parses and indexes movie transcripts into a Chroma vector store.
2. User asks a question — e.g., "What does Neo realize at the end of The Matrix?"
3. Relevant transcript chunks are retrieved via vector similarity.
4. These chunks + the user query are passed to the LLM (GPT) to generate the answer.
Answer is returned to the user via CLI or API.

---
🧪 Example Queries

Q: What is the name of the ship Morpheus commands?
A: The Nebuchadnezzar.

Q: What choice does Neo make at the end?
A: Neo chooses to sacrifice himself to save others, realizing his purpose.

---
🔮 Future Enhancements

Gradio/Streamlit UI for web-based interface
Hugging Face Space deployment
Add support for multi-language transcripts
Summarization & scene-level search

---
🤝 Contributing

Pull requests are welcome. Please open issues to discuss improvements or bugs.

---
📜 License

MIT License – feel free to use, fork, and modify.

📫 Contact

Made by Sandesh Mendan
Project inspired by movie nerds + LLMs ✨
