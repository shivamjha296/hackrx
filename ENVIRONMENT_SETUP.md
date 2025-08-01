# Environment Setup Guide

## 🔐 Setting up Environment Variables

This guide helps you set up environment variables for secure API key management before pushing to GitHub.

### Step 1: Copy Environment Template

```bash
cp .env.example .env
```

### Step 2: Edit .env File

Open the `.env` file and replace the placeholder values with your actual credentials:

```bash
# HackRX 5.0 Environment Variables
GEMINI_API_KEY=your_actual_gemini_api_key_here
BEARER_TOKEN=your_bearer_token_here
HUGGINGFACE_HUB_TOKEN=your_hf_token_here

# Server Configuration (optional to change)
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# Model Configuration (optional to change)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gemini-2.5-flash-lite

# Performance Settings (optional to change)
CHUNK_SIZE=1500
CHUNK_OVERLAP=250
RETRIEVAL_K=5
BATCH_SIZE=64
```

### Step 3: Get Your API Keys

#### Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to `GEMINI_API_KEY` in your `.env` file

#### HuggingFace Token (Optional)
1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a new token
3. Copy the token to `HUGGINGFACE_HUB_TOKEN` in your `.env` file

#### Bearer Token
- Use the provided hackathon token: `02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0`
- Or generate your own secure token for production use

### Step 4: Verify Setup

Run the validation script to ensure everything is configured correctly:

```bash
python validate.py
```

### Step 5: Test the Application

```bash
# Start the server
python start_server.py

# In another terminal, test the API
python test_api.py
```

## 🚨 Security Notes

### ✅ Safe for GitHub (these files will be committed):
- `.env.example` - Template with placeholder values
- `config.py` - Uses environment variables
- `app.py` - No hardcoded secrets
- `main.py` - Uses config instead of hardcoded keys

### ❌ NOT Safe for GitHub (these files are gitignored):
- `.env` - Contains your actual secrets
- `myenv/` or `myenv1/` - Virtual environment
- `pdf_documents/` - Your documents
- `*.log` - Log files

### 🔒 Additional Security Tips

1. **Never commit actual API keys**
2. **Use different keys for development/production**
3. **Rotate keys regularly**
4. **Use secure key management in production**

## 🚀 Ready for GitHub!

Once you've completed these steps:

1. Your secrets are in `.env` (gitignored)
2. Your code uses environment variables
3. You can safely push to GitHub
4. Other developers can use `.env.example` as a template

### Push to GitHub:

```bash
git add .
git commit -m "Add environment variable support for secure API key management"
git push origin main
```

## 🛠 Troubleshooting

### Issue: "GEMINI_API_KEY must be set in environment variables"
**Solution**: Make sure your `.env` file exists and contains a valid `GEMINI_API_KEY`

### Issue: "python-dotenv module not found"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: Environment variables not loading
**Solution**: Make sure `.env` file is in the same directory as your Python scripts

### Issue: Invalid API key error
**Solution**: Verify your Gemini API key is correct and active

---

**Your application is now secure and ready for GitHub! 🎉**
