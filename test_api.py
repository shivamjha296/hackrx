"""
Test script for HackRX 5.0 API
Tests the /hackrx/run endpoint with sample data
"""

import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
BASE_URL = "http://localhost:8000"
BEARER_TOKEN = os.getenv("BEARER_TOKEN", "your-bearer-token-here")

# Headers
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json", 
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

# Sample request data (as provided in the problem statement)
sample_request = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")

def test_hackrx_endpoint():
    """Test the main hackrx/run endpoint"""
    print("\nTesting hackrx/run endpoint...")
    print(f"Sending request with {len(sample_request['questions'])} questions...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=headers,
            json=sample_request,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            print("✅ Request successful!")
            result = response.json()
            
            print(f"\nReceived {len(result['answers'])} answers:")
            for i, answer in enumerate(result['answers'], 1):
                print(f"\nQ{i}: {sample_request['questions'][i-1]}")
                print(f"A{i}: {answer}")
            
            if 'metadata' in result:
                print(f"\nMetadata: {json.dumps(result['metadata'], indent=2)}")
                
        else:
            print(f"❌ Request failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out")
    except Exception as e:
        print(f"❌ Request error: {str(e)}")

def test_authentication():
    """Test authentication with invalid token"""
    print("\nTesting authentication...")
    
    invalid_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer invalid_token"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            headers=invalid_headers,
            json=sample_request
        )
        
        if response.status_code == 401:
            print("✅ Authentication test passed (correctly rejected invalid token)")
        else:
            print(f"❌ Authentication test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Authentication test error: {str(e)}")

def main():
    """Run all tests"""
    print("=" * 60)
    print("HackRX 5.0 API Test Suite")
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Test endpoints
    test_health_endpoint()
    test_authentication()
    test_hackrx_endpoint()
    
    print("\n" + "=" * 60)
    print("Test suite completed")

if __name__ == "__main__":
    main()
