# Max: AI-Powered Developer Portfolio Chatbot

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-%5E0.75-green) ![LangChain](https://img.shields.io/badge/LangChain-latest-orange) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## 🚀 Project Overview

**Max Portfolio Assistant** is an advanced, production-ready AI chatbot API designed to deliver *contextual, precise, and dynamic* answers about Nivakaran’s professional portfolio. Leveraging the power of retrieval-augmented generation (RAG) combined with vector-based semantic search, this system intelligently interprets user queries and fetches relevant information directly from portfolio documents, including detailed PDFs.

This is not just another chatbot — it’s a *smart assistant* engineered to showcase developer expertise with real-world, scalable architecture.

---

## 🔥 Why This Project Matters

- **Real-world AI Integration:** Implements cutting-edge LLM orchestration using LangChain with HuggingFace embeddings and Groq’s hosted large language model.
- **Production-Grade API:** Built on FastAPI with clear REST endpoints, session management, and CORS ready for seamless frontend integration.
- **Multi-turn Dialogue:** Maintains chat history context for fluid conversations — no robotic one-off answers.
- **Efficient Document Retrieval:** Processes and indexes large PDFs into vector embeddings enabling lightning-fast semantic search.
- **Clean Code & Logging:** Structured with robust error handling and logging — ready for maintenance and scaling.
- **Portfolio Showcase:** Serves as a unique interactive gateway into the developer’s skills, projects, and professional story.

---

## 🛠 Technical Highlights

| Feature                        | Details                                                                                  |
|--------------------------------|------------------------------------------------------------------------------------------|
| Language Model                 | Groq ChatGroq (Deepseek-R1-Distill-Llama-70b)                                            |
| Embeddings                     | HuggingFace `all-MiniLM-L6-v2` for semantic vector search                                |
| Vector Store                   | Chroma DB for fast, persistent vector retrieval                                          |
| Document Loader & Splitter     | `PyPDFLoader` + `RecursiveCharacterTextSplitter` for handling large portfolio PDFs       |
| API Framework                  | FastAPI with async support, CORS, and automatic Swagger docs                             |
| Session Management             | Session-aware chat history using LangChain's `ChatMessageHistory`                        |
| Environment & Config           | `.env` managed tokens and paths for secure, flexible deployment                          |
| Logging & Error Handling       | Python `logging` with clear error responses for production debugging                     |

---

## 📦 Installation & Setup

Setup Project
```bash
git clone https://github.com/yourusername/max-portfolio-assistant.git
cd max-portfolio-assistant
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create .env file with the following variables:
```bash
HF_TOKEN=your_huggingface_api_token
GROQ_API_KEY=your_groq_api_key
PDF_PATH=path/to/your/portfolio.pdf
HOST=0.0.0.0
PORT=5000

```

Run the server:
```bash 
uvicorn main:app --host $HOST --port $PORT --reload
```

## 🔍 How to Use
- Ask the assistant: Send POST requests to /ask with JSON body:
```bash

{
  "session_id": "unique-session-uuid",
  "question": "What are Nivakaran's main technical skills?"
}

```
- Get answers: The AI responds with precise, context-aware answers sourced directly from the portfolio.
- Maintain sessions: Use consistent session_id values to keep chat history context intact. 

## 🧠 Architecture Overview
1. PDF Processing: Portfolio PDF is loaded and split into manageable chunks.
2. Vectorization: Text chunks converted to semantic vectors via HuggingFace embeddings.
3. Indexing: Chroma database stores vectors for similarity search.
4. Query Handling: Incoming questions are reformulated based on chat history.
5. Retrieval & Generation: System retrieves relevant document chunks and generates an answer using Groq LLM.
6. Session Management: Multi-turn dialogue history tracked to ensure coherent conversations.

## 🎯 Impact & Use Cases
- Personal Branding: Transform your static portfolio into an interactive, AI-powered experience.
- Recruiter Friendly: Instant access to precise answers about skills and projects—no browsing required.
- Tech Demonstration: Showcases expertise in AI integration, API design, and modern NLP pipelines.
- Scalable Architecture: Easily extend to multiple domains or add new data sources.

## 📚 Technologies & Tools
- Python 3.8+
- FastAPI (ASGI web framework)
- LangChain (LLM orchestration)
- Groq ChatGroq (LLM inference)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Chroma Vector Database
- PyPDFLoader & RecursiveCharacterTextSplitter
- Pydantic for request validation
- Uvicorn ASGI server
- Python Logging

## 🤝 Contributing
Contributions and improvements are welcome! Feel free to open issues or submit PRs for:
- Adding new document types
- Enhancing conversation flow
- Improving deployment (Docker/K8s)
- Optimizing vector search performance


## ⚡ Final Notes
Max Portfolio Assistant is not just a chatbot, it’s a showcase of how to leverage modern AI, NLP, and backend engineering skills to create real, usable developer portfolio experiences that recruiters notice and remember.

