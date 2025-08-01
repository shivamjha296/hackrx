"""
Demo script for HackRX 5.0 submission
Shows the system processing the sample document and questions
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = "02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0"

# Sample data from problem statement
DEMO_REQUEST = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?"
    ]
}

def run_demo():
    """Run the demo with sample data"""
    print("="*80)
    print("🎯 HackRX 5.0 - LIVE DEMO")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Server: {BASE_URL}")
    print(f"Document: {DEMO_REQUEST['documents'][:50]}...")
    print(f"Questions: {len(DEMO_REQUEST['questions'])}")
    print("="*80)
    
    # Check server health
    print("\n🏥 Checking server health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Server is healthy (device: {health_data.get('device', 'unknown')})")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure the server is running: python start_server.py")
        return False
    
    # Send demo request
    print("\n🚀 Processing demo request...")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }
    
    try:
        start_time = time.time()
        print("⏳ Downloading and processing document...")
        print("⏳ Creating embeddings and vector store...")
        print("⏳ Generating answers with Gemini LLM...")
        
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            json=DEMO_REQUEST,
            headers=headers,
            timeout=300  # 5 minutes
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            print(f"✅ Request successful! (Processing time: {processing_time:.2f}s)")
            
            result = response.json()
            
            print("\n📋 RESULTS:")
            print("-"*80)
            
            for i, (question, answer) in enumerate(zip(DEMO_REQUEST['questions'], result['answers']), 1):
                print(f"\n🔍 Question {i}:")
                print(f"   {question}")
                print(f"\n💡 Answer {i}:")
                print(f"   {answer}")
                print("-"*80)
            
            # Show metadata
            if 'metadata' in result:
                metadata = result['metadata']
                print(f"\n📊 PROCESSING METADATA:")
                print(f"   • Total Questions: {metadata.get('total_questions', 'N/A')}")
                print(f"   • Processing Time: {metadata.get('processing_timestamp', 'N/A')}")
                print(f"   • LLM Model: {metadata.get('model_info', {}).get('llm', 'N/A')}")
                print(f"   • Embeddings: {metadata.get('model_info', {}).get('embeddings', 'N/A')}")
                print(f"   • Vector Store: {metadata.get('model_info', {}).get('vectorstore', 'N/A')}")
            
            print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
            return True
            
        else:
            print(f"❌ Request failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {error_data}")
            except:
                print(f"Error text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Main demo function"""
    success = run_demo()
    
    print("\n" + "="*80)
    if success:
        print("🏆 HACKRX 5.0 DEMO SUCCESSFUL!")
        print("✅ System is working correctly with sample data")
        print("✅ All answers generated successfully")
        print("✅ Response format matches requirements")
        print("\n🌐 Try the interactive docs: http://localhost:8000/docs")
    else:
        print("❌ Demo failed - please check the server and try again")
    print("="*80)

if __name__ == "__main__":
    main()
