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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.tools import Tool


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 5000))
PDF_PATH = os.getenv("PDF_PATH", "nivakaran.pdf")

# Validate environment variables
if not all([HF_TOKEN, GROQ_API_KEY, PDF_PATH]):
    logger.error("Missing required environment variables")
    raise RuntimeError("Environment variables not set")

# Initialize FastAPI app
app = FastAPI(
    title="Portfolio API",
    description="API for Nivakaran's portfolio",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize RAG components
embeddings = HuggingFaceEmbeddings(model_name="./local_model")
llm = ChatGroq(model_name="Deepseek-R1-Distill-Llama-70b")
session_store = {}

def process_pdf(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./portfolio.db"
        )
        logger.info(f"PDF {file_path} processed successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        raise RuntimeError("PDF processing failed")

# Initialize vectorstore
try:
    vectorstore = process_pdf(PDF_PATH)
    retriever = vectorstore.as_retriever()
    logger.info("Vectorstore initialized successfully")
except Exception as e:
    logger.error(f"Vectorstore initialization failed: {str(e)}")
    raise RuntimeError("Vectorstore initialization failed")


class QuestionRequest(BaseModel):
    session_id: str
    question: str

class QuestionResponse(BaseModel):
    answer: str


@app.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask the portfolio assistant",
    description="Submit a question to get a reply from Max, the portfolio chatbot."
)
async def ask_question(request: QuestionRequest):
    session_id = request.session_id
    question = request.question
    logger.info(f"Received question for session {session_id}: {question}")

    try:
        if session_id not in session_store:
            session_store[session_id] = {
                "history": ChatMessageHistory(),
                "retriever": retriever
            }

        session = session_store[session_id]
        history = session["history"]
        last_messages = history.messages[-6:]

        # RAG processing
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Rephrase questions considering chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, session["retriever"], contextualize_q_prompt
        )

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

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Get and process response
        result = rag_chain.invoke({
            "input": question,
            "chat_history": last_messages
        })
        raw_answer = result["answer"]

        # Remove <think>...</think> block from answer
        cleaned_answer = re.sub(r"<think>.*?</think>\s*", "", raw_answer, flags=re.DOTALL).strip()

        # Update history
        history.add_user_message(question)
        history.add_ai_message(cleaned_answer)

        logger.info(f"Cleaned response for session {session_id}: {cleaned_answer[:100]}...")
        return QuestionResponse(answer=cleaned_answer)

    except Exception as e:
        logger.error(f"Error processing question for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Portfolio API",
        "endpoints": {
            "portfolio_assistant": "/ask",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)