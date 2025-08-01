"""
Validation script for HackRX 5.0 solution
Checks all dependencies and configurations
"""

import sys
import os
from datetime import datetime

def check_imports():
    """Check all required imports"""
    print("🔍 Checking imports...")
    
    try:
        import fastapi
        print("✅ FastAPI imported")
        
        import uvicorn
        print("✅ Uvicorn imported")
        
        import google.generativeai as genai
        print("✅ Google Generative AI imported")
        
        from langchain_community.document_loaders import PyPDFLoader
        print("✅ LangChain community imported")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ LangChain HuggingFace imported")
        
        import torch
        print(f"✅ PyTorch imported (device: {'cuda' if torch.cuda.is_available() else 'cpu'})")
        
        import faiss
        print("✅ FAISS imported")
        
        import requests
        print("✅ Requests imported")
        
        from docx import Document
        print("✅ python-docx imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def check_configuration():
    """Check configuration settings"""
    print("\n🔧 Checking configuration...")
    
    try:
        from config import Config
        Config.validate_config()
        print("✅ Configuration validated")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_api_structure():
    """Check that the API app can be imported"""
    print("\n🌐 Checking API structure...")
    
    try:
        from app import app
        print("✅ FastAPI app imported successfully")
        
        # Check endpoints
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/health", "/hackrx/run"]
        
        for route in expected_routes:
            if route in routes:
                print(f"✅ Route {route} found")
            else:
                print(f"❌ Route {route} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ API structure error: {e}")
        return False

def check_directories():
    """Check required directories exist"""
    print("\n📁 Checking directories...")
    
    required_dirs = ["pdf_documents"]
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Directory {dir_name} exists")
        else:
            os.makedirs(dir_name)
            print(f"✅ Created directory {dir_name}")
    
    return True

def run_validation():
    """Run all validation checks"""
    print("=" * 60)
    print("HackRX 5.0 - Solution Validation")  
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    checks = [
        ("Import Check", check_imports),
        ("Configuration Check", check_configuration),
        ("API Structure Check", check_api_structure),
        ("Directory Check", check_directories)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Ready for HackRX 5.0 submission!")
        print("\nNext steps:")
        print("1. Run: python start_server.py")
        print("2. Test: python test_api.py")
        print("3. Access docs: http://localhost:8000/docs")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
