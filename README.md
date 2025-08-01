# HackRX 5.0 - LLM-Powered Intelligent Query-Retrieval System

## 🏆 Competition Solution

This is a comprehensive solution for the HackRX 5.0 hackathon challenge, implementing an LLM-powered intelligent query-retrieval system that processes large documents and makes contextual decisions for insurance, legal, HR, and compliance domains.

## 🚀 Features

### Core Capabilities
- **Multi-format Document Processing**: Supports PDF, DOCX, and email documents
- **Semantic Search**: FAISS-based vector embeddings for efficient document retrieval  
- **LLM Integration**: Google Gemini 2.5 Flash Lite for intelligent response generation
- **RESTful API**: FastAPI-based service with proper authentication
- **Contextual Q&A**: Handles complex queries with explainable reasoning
- **Real-time Processing**: Optimized for low-latency responses

### Technical Specifications
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS for semantic similarity search
- **LLM**: Google Gemini 2.5 Flash Lite
- **Framework**: FastAPI with async support
- **Authentication**: Bearer token-based security

## 📁 Project Structure

```
hackrx/
├── app.py                 # Main FastAPI application
├── main.py               # Original chatbot with enhancements
├── config.py             # Configuration settings
├── test_api.py           # API testing script
├── start_server.py       # Server startup script
├── requirements.txt      # Python dependencies
├── README.md            # This documentation
├── myenv/               # Python virtual environment
├── pdf_documents/       # Document storage directory
└── pdf_vectorstore/     # Vector store cache
```

## 🛠 Installation & Setup

### Prerequisites
- Python 3.8+
- Windows (PowerShell) environment
- Valid Google Gemini API key

### 1. Environment Setup
```powershell
# Activate virtual environment
myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Set up your environment variables:

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your actual credentials:**
   ```bash
   GEMINI_API_KEY=your_actual_gemini_api_key_here
   BEARER_TOKEN=02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0
   ```

3. **Get your Gemini API key:**
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file

### 3. Quick Start
```powershell
# Method 1: Use startup script
python start_server.py

# Method 2: Direct uvicorn
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 🌐 API Documentation

### Base URL
```
http://localhost:8000
```

### Authentication
```
Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0
```
*Note: This token is now loaded from environment variables for security*

### Endpoints

#### 1. Health Check
```http
GET /health
```

#### 2. Main Processing Endpoint
```http
POST /hackrx/run
Content-Type: application/json
Authorization: Bearer {token}
```

Request Body:
```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What are the coverage limits?"
  ]
}
```

Response:
```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date.",
    "The coverage limit is set at $100,000 per incident with annual aggregate limits."
  ],
  "metadata": {
    "document_url": "https://example.com/document.pdf",
    "total_questions": 2,
    "processing_timestamp": "2025-08-01T10:30:00",
    "model_info": {
      "llm": "gemini-2.5-flash-lite",
      "embeddings": "sentence-transformers/all-MiniLM-L6-v2",
      "vectorstore": "FAISS"
    }
  }
}
```

#### 3. Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

### Automated Testing
```powershell
# Run comprehensive test suite
python test_api.py
```

## 🎯 System Architecture

### Workflow Pipeline
```
1. Document Input (PDF/DOCX/Email URL)
    ↓
2. Document Download & Processing
    ↓
3. Text Extraction & Chunking
    ↓
4. Embedding Generation (Sentence Transformers)
    ↓
5. Vector Store Creation (FAISS)
    ↓
6. Query Processing (LLM + Retrieval)
    ↓
7. Structured JSON Response
```

## 🏅 Evaluation Criteria Compliance

### ✅ Accuracy
- Domain-specific prompt engineering
- Multi-document retrieval with relevance scoring
- Structured response format ensuring completeness

### ✅ Token Efficiency
- Optimized chunk sizes for context windows
- Batch processing to minimize API calls
- Efficient embedding model selection

### ✅ Latency
- FAISS vector store for fast retrieval
- Async processing capabilities
- Memory management and cleanup

### ✅ Reusability
- Modular architecture with clear separation
- Configuration-driven design
- Docker-ready containerization support

### ✅ Explainability
- Source document tracking
- Confidence scoring in retrieval
- Metadata inclusion in responses

## 🚀 Quick Start Guide

1. **Activate Environment**: `myenv\Scripts\activate`
2. **Start Server**: `python start_server.py`
3. **Test API**: `python test_api.py`
4. **Access Docs**: http://localhost:8000/docs

**Ready for HackRX 5.0 submission! 🚀**