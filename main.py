"""
HackRX 5.0 - LLM-Powered Intelligent Query-Retrieval System
Original chatbot functionality with enhanced features for the hackathon
"""

import os
import shutil
import torch
from tqdm import tqdm  # Fixed import for general Python use
import gc
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from typing import Any, List, Optional, Mapping
from pydantic import Field
import google.generativeai as genai

import json
from datetime import datetime
import requests
from pathlib import Path

# Import configuration
from config import Config

# --- Helper Classes ---

# Define FeedbackStore class
class FeedbackStore:
    def __init__(self, filename="feedback_data.json"):
        self.filename = filename
        self.feedback_data = self.load_feedback()

    def load_feedback(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_feedback(self, question, answer, is_helpful, improvement_feedback=None):
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'is_helpful': is_helpful,
            'improvement_feedback': improvement_feedback
        }
        self.feedback_data.append(feedback_entry)

        with open(self.filename, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)

# Define GeminiLLM class
class GeminiLLM(LLM):
    model_name: str = "gemini-2.5-flash-lite"
    model: Any = Field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = genai.GenerativeModel(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[Any] = None) -> str:
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name}

# --- SubtitleChatbot Class ---

class SubtitleChatbot:
    def __init__(self, pdf_dir='pdf_documents'):
        self.pdf_dir = pdf_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        try:
            import tensorflow as tf
            print(f"TensorFlow version: {tf.__version__}")
            import keras
            print(f"Keras version: {keras.__version__}")
        except ImportError:
            print("TensorFlow/Keras not properly installed")


    def load_documents(self):
        """Load all PDF files from the directory with progress bar."""
        print("Loading documents...")
        files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        documents = []

        if not files:
            print(f"No PDF files found in '{self.pdf_dir}'. Please add files to this directory.")
            return documents

        # Fixed import from tqdm.notebook to tqdm for general Python use
        from tqdm import tqdm
        for file in tqdm(files, desc="Loading PDF files"):
            try:
                loader = PyPDFLoader(os.path.join(self.pdf_dir, file))
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")

        print(f"Loaded {len(documents)} documents")
        return documents


    def split_texts(self, documents):
        """Split documents into smaller chunks with progress tracking."""
        print("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            length_function=len
        )

        texts = []
        if not documents:
            print("No documents to split.")
            return texts

        # Fixed import from tqdm.notebook to tqdm for general Python use
        from tqdm import tqdm
        for doc in tqdm(documents, desc="Splitting documents"):
            texts.extend(text_splitter.split_documents([doc]))

        print(f"Created {len(texts)} text chunks")
        return texts


    def create_embeddings(self, texts):
        """Create embeddings with batch processing and memory management."""
        print("Creating embeddings...")

        if not texts:
             print("No texts to create embeddings from.")
             return None

        from langchain_huggingface import HuggingFaceEndpointEmbeddings
        model = Config.EMBEDDING_MODEL
        embeddings = HuggingFaceEndpointEmbeddings(
            model=model,
            huggingfacehub_api_token=Config.HUGGINGFACE_API_TOKEN,
        )

        batch_size = 64
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        vectorstore = None
        # Fixed import from tqdm.notebook to tqdm for general Python use
        from tqdm import tqdm
        for i, batch in enumerate(tqdm(batches, desc="Creating embeddings")):
            batch_vectorstore = FAISS.from_documents(batch, embeddings)

            if vectorstore is None:
                vectorstore = batch_vectorstore
            else:
                vectorstore.merge_from(batch_vectorstore)

            if self.device == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

            if (i + 1) % 10 == 0:
                vectorstore.save_local("pdf_vectorstore_checkpoint")

        print("Embeddings creation completed")
        return vectorstore

    def setup_chatbot(self, vectorstore):
        """Set up the conversational chain with Gemini."""
        print("Setting up chatbot...")

        if vectorstore is None:
            print("Cannot setup chatbot, vectorstore is None.")
            return None

        llm = GeminiLLM()
        feedback_store = FeedbackStore()

        recent_feedback = feedback_store.feedback_data[-5:] if feedback_store.feedback_data else []
        feedback_prompt = "\n".join([
            f"Previous improvement feedback: {f['improvement_feedback']}"
            for f in recent_feedback if not f['is_helpful'] and f['improvement_feedback']
        ])

        prompt_template = f"""You are a helpful AI assistant.

Previous feedback on similar responses suggests the following improvements:
{feedback_prompt}

Based on the provided context, answer the question with a single, comprehensive sentence that captures all the essential information.

Context: {{context}}

Question: {{question}}

Provide your answer as one complete, well-structured sentence that includes all relevant details, examples, and key information from the context. Make it clear, comprehensive, and direct.

Example format: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."

Answer:"""

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            ),
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )},
            verbose=True
        )

        return qa_chain

    def load_model(self, path="pdf_vectorstore"):
        """Load a previously trained model."""
        print(f"Loading trained model from {path}...")
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': self.device}
            )
            vectorstore = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
            qa_chain = self.setup_chatbot(vectorstore)
            print("Model loaded successfully!")
            return qa_chain
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def train(self):
        """Train the chatbot with progress tracking and checkpoints."""
        try:
            documents = self.load_documents()
            if not documents:
                 print("No documents loaded, cannot proceed with training.")
                 return None

            texts = self.split_texts(documents)
            if not texts:
                 print("No texts created from documents, cannot proceed with training.")
                 return None

            vectorstore = self.create_embeddings(texts)
            if vectorstore is None:
                print("Vector store creation failed, cannot proceed with training.")
                return None

            print("Saving final vector store...")
            vectorstore.save_local("pdf_vectorstore")

            qa_chain = self.setup_chatbot(vectorstore)
            print("Training completed successfully!")
            return qa_chain

        except Exception as e:
            print(f"Error during training: {str(e)}")
            if os.path.exists("pdf_vectorstore_checkpoint"):
                print("Attempting to load from last checkpoint due to training error...")
                return self.load_model("pdf_vectorstore_checkpoint")
            else:
                 print("No checkpoint found to load from. Training failed.")
            raise e

# --- Chat Interface ---

def chat_interface(qa_chain):
    """Simple chat interface for interacting with the bot."""
    if qa_chain is None:
        print("Chatbot is not initialized. Cannot start chat interface.")
        return

    print("\nChatbot is ready! Type 'quit' to exit.")
    while True:
        question = input("\nYou: ")
        if question.lower() == 'quit':
            break

        try:
            result = qa_chain({"question": question})
            print("\nBot:", result["answer"])

            if "source_documents" in result and result["source_documents"]:
                print("\nSource Documents:")
                sources = set()
                for doc in result["source_documents"]:
                     source_info = doc.metadata.get('source', 'N/A')
                     page_info = doc.metadata.get('page', 'N/A')
                     sources.add(f"📄 Source: {source_info}, Page: {page_info}")

                for source in sources:
                     print(source)

            elif "source_documents" in result and not result["source_documents"]:
                print("\nNo specific source documents found for this answer.")

        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try asking your question in a different way.")


# --- Main Execution ---

if __name__ == "__main__":
    # Use configuration from config.py
    genai.configure(api_key=Config.GEMINI_API_KEY)

    if not Config.GEMINI_API_KEY or Config.GEMINI_API_KEY == "your-gemini-api-key-here":
        print("Please set your Gemini API key in the .env file first!")
        print("Copy .env.example to .env and add your actual API key")
    else:
        print("Initializing chatbot...")
        pdf_directory = 'pdf_documents'

        if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory)
            print(f"Directory '{pdf_directory}' created.")
        else:
            print(f"Directory '{pdf_directory}' already exists.")

        print(f"\n--- IMPORTANT ---")
        print(f"Please ensure you have placed your sample PDF files into the '{pdf_directory}' directory.")
        print(f"You can ignore the following steps if the directory is empty and you don't have files yet.")
        print(f"-----------------\n")


        pdf_vectorstore_path = "pdf_vectorstore"
        pdf_vectorstore_checkpoint_path = "pdf_vectorstore_checkpoint"

        chatbot = SubtitleChatbot(pdf_dir=pdf_directory)

        qa_chain = None
        if os.path.exists(pdf_vectorstore_path):
            qa_chain = chatbot.load_model(path=pdf_vectorstore_path)
        else:
            if os.path.exists(pdf_vectorstore_checkpoint_path):
                 print("Final vector store not found, attempting to load from checkpoint...")
                 qa_chain = chatbot.load_model(path=pdf_vectorstore_checkpoint_path)
                 if qa_chain:
                     print("Loaded from checkpoint. You may want to re-run training to complete.")
                 else:
                     print("Failed to load from checkpoint. Starting full training.")
                     qa_chain = chatbot.train()
            else:
                print("No existing vector store or checkpoint found. Starting full training.")
                qa_chain = chatbot.train()

        if qa_chain:
            chat_interface(qa_chain)
        else:
            print("Failed to initialize chatbot. Please check the errors and ensure you have PDFs in the directory.")