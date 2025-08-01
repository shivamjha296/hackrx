"""
Startup script for HackRX 5.0 API Server
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_environment():
    """Check if we're in the correct Python environment"""
    python_path = sys.executable
    print(f"Using Python: {python_path}")
    
    # Check if we're in the virtual environment
    if 'myenv' not in python_path:
        print("Warning: Not using the virtual environment")
        print("Please activate the virtual environment first:")
        print("myenv\\Scripts\\activate")
        return False
    
    return True

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("Starting HackRX 5.0 API Server...")
    print("Server will be available at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

def main():
    """Main startup function"""
    print("=" * 60)
    print("HackRX 5.0 - Intelligent Query-Retrieval System")
    print("Starting up...")
    print("=" * 60)
    
    # Check environment
    if not check_python_environment():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
