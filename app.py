import os
import io
import json
import re
import logging
import tempfile
import base64
from uuid import uuid4
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from pymongo import MongoClient


# Alternative PDF libraries for fallback
try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False
    
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URL = os.getenv("MONGODB_URL")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "test")

# Parse collections as a list from comma-separated string in .env
collections_env = os.getenv("MONGODB_COLLECTION", "blogs")
MONGODB_COLLECTIONS = [col.strip() for col in collections_env.split(",") if col.strip()]

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
PDF_PATH = os.getenv("PDF_PATH", "./nivakaran.pdf")


# Validate environment variables
if not all([HF_TOKEN, GROQ_API_KEY, PDF_PATH, MONGODB_URL]):
    logger.error("Missing required environment variables")
    raise RuntimeError("Environment variables not set. Please check HF_TOKEN, GROQ_API_KEY, PDF_PATH, and MONGODB_URL")


# Initialize MongoDB client
try:
    mongo_client = MongoClient(MONGODB_URL)
    mongo_client.admin.command('ping')
    logger.info("MongoDB connection successful")
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    raise RuntimeError("MongoDB connection failed")


# Initialize FastAPI app
app = FastAPI(
    title="Portfolio API",
    description="Chatbot for Nivakaran's Portfolio.",
    version="1.0.0",
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# Initialize RAG components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model_name="openai/gpt-oss-20b")


def extract_text_with_pypdf(file_path: str) -> List[Document]:
    """Extract text using pypdf library directly"""
    try:
        reader = PdfReader(file_path)
        documents = []
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():  # Only add non-empty pages
                doc = Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                )
                documents.append(doc)
        
        logger.info(f"pypdf extracted text from {len(documents)} pages")
        return documents
    except Exception as e:
        logger.error(f"pypdf extraction failed: {str(e)}")
        return []


def extract_text_with_pymupdf(file_path: str) -> List[Document]:
    """Extract text using PyMuPDF (fitz) library - often better for complex PDFs"""
    try:
        doc = fitz.open(file_path)
        documents = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            if text.strip():  # Only add non-empty pages
                document = Document(
                    page_content=text,
                    metadata={"source": file_path, "page": page_num}
                )
                documents.append(document)
        
        doc.close()
        logger.info(f"PyMuPDF extracted text from {len(documents)} pages")
        return documents
    except Exception as e:
        logger.error(f"PyMuPDF extraction failed: {str(e)}")
        return []


def process_pdf(file_path: str):
    """Process PDF with multiple fallback methods for robust text extraction"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")
        
        logger.info(f"Processing PDF from: {file_path}")
        documents = []
        
        # Method 1: Try LangChain's PyPDFLoader (uses pypdf internally)
        try:
            logger.info("Attempting extraction with PyPDFLoader...")
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if documents and any(doc.page_content.strip() for doc in documents):
                logger.info(f"PyPDFLoader successfully loaded {len(documents)} pages")
            else:
                documents = []
                logger.warning("PyPDFLoader returned empty documents")
        except Exception as e:
            logger.warning(f"PyPDFLoader failed: {str(e)}")
        
        # Method 2: Try direct pypdf if available and previous method failed
        if not documents and PYPDF_AVAILABLE:
            logger.info("Attempting extraction with pypdf directly...")
            documents = extract_text_with_pypdf(file_path)
        
        # Method 3: Try PyMuPDF as fallback (often best for complex PDFs)
        if not documents and PYMUPDF_AVAILABLE:
            logger.info("Attempting extraction with PyMuPDF (fitz)...")
            documents = extract_text_with_pymupdf(file_path)
        
        # Validate that documents were loaded
        if not documents:
            raise ValueError(
                "Failed to extract text from PDF with all available methods. "
                "The PDF might be:\n"
                "1. Empty or corrupted\n"
                "2. Password-protected\n"
                "3. Scanned images without OCR (consider using pytesseract)\n"
                "4. Using unsupported encryption"
            )
        
        # Check if any text was actually extracted
        total_text = "".join([doc.page_content for doc in documents])
        if not total_text.strip():
            raise ValueError("No text content found in PDF. It may contain only images.")
        
        logger.info(f"Successfully extracted {len(total_text)} characters from {len(documents)} pages")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=5000,
            chunk_overlap=500,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # Filter out empty chunks
        splits = [doc for doc in splits if doc.page_content.strip()]
        
        if not splits:
            raise ValueError("Text splitting resulted in zero valid chunks.")
        
        logger.info(f"Created {len(splits)} text chunks for vectorization")
        
        # Create vectorstore
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./portfolio.db"
        )
        
        logger.info("Vectorstore created successfully")
        return vectorstore
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        raise RuntimeError(f"PDF file not found: {str(e)}")
    except ValueError as e:
        logger.error(f"Invalid PDF content: {str(e)}")
        raise RuntimeError(f"PDF processing failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error processing PDF: {str(e)}", exc_info=True)
        raise RuntimeError(f"PDF processing failed: {str(e)}")


def get_session_histories(session_id: str) -> List[MongoDBChatMessageHistory]:
    """Get list of MongoDB chat message histories for a session from all collections"""
    histories = []
    for col in MONGODB_COLLECTIONS:
        history = MongoDBChatMessageHistory(
            connection_string=MONGODB_URL,
            session_id=session_id,
            database_name=MONGODB_DATABASE,
            collection_name=col,
            create_index=True
        )
        histories.append(history)
    return histories


def merge_histories(histories: List[MongoDBChatMessageHistory]) -> List:
    """Merge messages from multiple histories sorted by creation time if available"""
    all_messages = []
    for history in histories:
        all_messages.extend(history.messages)
    # Sort by timestamp or insertion order if 'created_at' attribute exists
    all_messages.sort(key=lambda msg: getattr(msg, 'created_at', 0))
    return all_messages


# Initialize vectorstore
try:
    logger.info(f"Initializing vectorstore from PDF: {PDF_PATH}")
    vectorstore = process_pdf(PDF_PATH)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    logger.info("Vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Vectorstore initialization failed: {str(e)}")
    logger.error("\nTroubleshooting steps:")
    logger.error("1. Verify PDF file exists at the specified path")
    logger.error("2. Ensure PDF contains extractable text (not just scanned images)")
    logger.error("3. Check if PDF is password-protected")
    logger.error("4. Try opening the PDF manually to verify it's not corrupted")
    logger.error("\nInstall additional libraries for better PDF support:")
    logger.error("  pip install pypdf pymupdf")
    raise RuntimeError(f"Vectorstore initialization failed: {str(e)}")


class QuestionRequest(BaseModel):
    session_id: str
    question: str


class QuestionResponse(BaseModel):
    answer: str


class SessionHistoryRequest(BaseModel):
    session_id: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    message_count: int
    messages: List[dict]


@app.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask the Nivakaran's portfolio assistant",
    description="Submit a question to learn about nivakaran's projects, and so on."
)
async def ask_question(request: QuestionRequest):
    """Handle question and maintain chat history in MongoDB across multiple collections"""
    session_id = request.session_id
    question = request.question
    logger.info(f"Received question for session {session_id}: {question}")

    try:
        # Get chat histories from all collections
        histories = get_session_histories(session_id)
        all_messages = merge_histories(histories)

        # Keep last 6 messages for chat history context
        last_messages = all_messages[-6:] if len(all_messages) > 6 else all_messages

        # Extract full session context text from all messages
        session_context_text = "\n".join(
            [msg.content for msg in all_messages if hasattr(msg, "content") and msg.content.strip()]
        )

        # System prompt now expects {context} as input variable
        system_prompt = """You are Max, a friendly and professional chatbot designed to 
assist visitors to Nivakaran’s portfolio website. Your primary goal 
is to provide accurate, clear, and helpful information about Nivakaran, based 
on the following context:

{context}

Your responses should be:
1. Informative and relevant, directly addressing the visitor’s questions about Nivakaran’s skills, 
projects, experience, and background.
2. Concise but thorough enough to give visitors a clear understanding of Nivakaran’s expertise.
3. Engaging and approachable, maintaining a professional yet conversational tone.
4. Honest about what is available in the provided context; if you don’t know an answer, politely 
say so and suggest the visitor explore other sections of the portfolio or contact Nivakaran directly.
5. Focused on helping visitors understand Nivakaran’s capabilities and what makes him stand out 
as a developer and professional.
6. Ready to provide examples, explanations, or links to portfolio projects when relevant.

Avoid providing generic or unrelated information. Always tailor your answers to 
highlight Nivakaran’s strengths and the unique value he brings.
"""

        # Create ChatPromptTemplate with variables {context} and {input}, plus chat_history placeholder
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, ChatPromptTemplate.from_messages([
                ("system", "Rephrase the user's question considering the chat history to provide better context."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
        )

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Invoke RAG chain passing question, full context text, and last 6 chat messages
        result = rag_chain.invoke({
            "input": question,
            "context": session_context_text,
            "chat_history": last_messages
        })
        raw_answer = result["answer"]

        # Clean answer by removing any <think>...</think> blocks
        cleaned_answer = re.sub(r"<think>.*?</think>\s*", "", raw_answer, flags=re.DOTALL).strip()

        # Add user question and AI response to all histories (all collections)
        for history in histories:
            history.add_user_message(question)
            history.add_ai_message(cleaned_answer)

        logger.info(f"Response saved to MongoDB for session {session_id}")
        return QuestionResponse(answer=cleaned_answer)

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/history", response_model=SessionHistoryResponse)
async def get_history(request: SessionHistoryRequest):
    """Retrieve chat history for a session from all collections"""
    try:
        histories = get_session_histories(request.session_id)
        all_messages = merge_histories(histories)
        messages_dict = [{"type": msg.type, "content": msg.content} for msg in all_messages]
        return SessionHistoryResponse(
            session_id=request.session_id,
            message_count=len(all_messages),
            messages=messages_dict
        )
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve history: {str(e)}")


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for a session from all collections"""
    try:
        histories = get_session_histories(session_id)
        for history in histories:
            history.clear()
        logger.info(f"Cleared history for session {session_id}")
        return {"message": f"History cleared for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear history: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        mongo_client.admin.command('ping')
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"disconnected: {str(e)}"
    
    return {
        "status": "healthy",
        "app": "Nivakaran's Portfolio Assistant",
        "mongodb": mongo_status,
        "vectorstore": "initialized" if vectorstore else "not initialized",
        "pdf_libraries": {
            "pypdf": PYPDF_AVAILABLE,
            "pymupdf": PYMUPDF_AVAILABLE
        }
    }


@app.get("/")
async def root():
    return {
        "message": "Welcome to Nivakaran's Portfolio API",
        "description": "Learn about reforestation, tree planting, and environmental conservation",
        "endpoints": {
            "ask_question": "/ask",
            "get_history": "/history",
            "clear_history": "/history/{session_id}",
            "health_check": "/health",
            "documentation": "/docs"
        }
    }


@app.on_event("shutdown")
async def shutdown_event():
    """Close MongoDB connection"""
    mongo_client.close()
    logger.info("MongoDB connection closed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
