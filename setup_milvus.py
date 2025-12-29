"""
Quick setup script to configure Milvus/Zilliz Cloud for your RAG chatbot
Run this script after setting up your credentials in .env file
"""
import os
import sys
from pathlib import Path


def check_env_file():
    """Check if .env file exists and has required variables"""
    print("\n" + "="*70)
    print("STEP 1: Checking .env file")
    print("="*70)
    
    env_path = Path(__file__).parent / '.env'
    
    if not env_path.exists():
        print("✗ .env file not found!")
        print("\nPlease create a .env file with the following content:")
        print("-" * 70)
        print("""
# Vector Store Configuration
VECTOR_STORE=milvus

# Zilliz Cloud Configuration
ZILLIZ_CLOUD_URI=https://your-cluster.api.gcp-us-west1.zillizcloud.com
ZILLIZ_CLOUD_TOKEN=your_api_token_here
MILVUS_COLLECTION_NAME=tech_chatbot_collection
MILVUS_EMBEDDING_DIM=896

# Other required configurations (keep existing values)
QWEN_EMBEDDING_MODEL=Qwen/Qwen2.5-0.5B-Instruct
GEMINI_API_KEY=your_gemini_api_key
DEVICE=cpu
        """)
        print("-" * 70)
        return False
    
    print("✓ .env file found")
    
    # Check for required variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        'ZILLIZ_CLOUD_URI': 'Zilliz Cloud endpoint URI',
        'ZILLIZ_CLOUD_TOKEN': 'Zilliz Cloud API token',
        'MILVUS_COLLECTION_NAME': 'Collection name',
        'MILVUS_EMBEDDING_DIM': 'Embedding dimension',
        'QWEN_EMBEDDING_MODEL': 'Qwen model name',
        'GEMINI_API_KEY': 'Gemini API key'
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"  - {var}: {description}")
            print(f"✗ Missing: {var}")
        else:
            # Mask sensitive values
            if 'TOKEN' in var or 'KEY' in var:
                display_value = value[:10] + '...' if len(value) > 10 else '***'
            else:
                display_value = value
            print(f"✓ Found: {var} = {display_value}")
    
    if missing_vars:
        print("\n✗ Missing required environment variables:")
        for var in missing_vars:
            print(var)
        return False
    
    # Note: VECTOR_STORE variable is no longer used (Milvus only)
    print("\n✓ Using Milvus/Zilliz Cloud for vector storage")
    
    return True


def test_imports():
    """Test if all required packages are installed"""
    print("\n" + "="*70)
    print("STEP 2: Checking Python dependencies")
    print("="*70)
    
    required_packages = {
        'pymilvus': 'Milvus client',
        'langchain': 'LangChain framework',
        'langchain_core': 'LangChain core',
        'transformers': 'Hugging Face Transformers',
        'torch': 'PyTorch',
        'google.generativeai': 'Google Gemini API'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {package}: {description}")
        except ImportError:
            print(f"✗ {package}: {description} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n✗ Missing packages. Install with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✓ All required packages installed")
    return True


def run_verification():
    """Run the verification script"""
    print("\n" + "="*70)
    print("STEP 3: Running verification tests")
    print("="*70)
    
    print("\nRunning embedding and connection tests...")
    print("This may take a minute...\n")
    
    # Change to src directory
    src_dir = Path(__file__).parent / 'src'
    os.chdir(src_dir)
    
    # Run test script
    result = os.system(f'{sys.executable} -m rag.test_embeddings')
    
    return result == 0


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    
    print("""
1. Review your Zilliz Cloud configuration above

2. Ingest your data to Milvus:
   cd src
   python -m rag.data_ingestion --vector-store milvus

3. Test the RAG pipeline:
   cd src
   python -m rag.test_pipeline

4. Run the Streamlit chatbot:
   cd src
   streamlit run rag/streamlit_app.py

For detailed documentation, see: MILVUS_SETUP.md
    """)
    print("="*70 + "\n")


def main():
    """Main setup flow"""
    print("\n" + "="*70)
    print("MILVUS/ZILLIZ CLOUD SETUP")
    print("="*70)
    print("\nThis script will help you set up Milvus on Zilliz Cloud")
    print("for your RAG chatbot.")
    
    # Step 1: Check .env file
    if not check_env_file():
        print("\n✗ Setup incomplete. Please fix the issues above and run again.")
        return False
    
    # Step 2: Test imports
    if not test_imports():
        print("\n✗ Setup incomplete. Please install missing packages and run again.")
        return False
    
    # Step 3: Run verification
    try:
        verification_passed = run_verification()
    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        verification_passed = False
    
    # Print next steps
    print_next_steps()
    
    if verification_passed:
        print("✓ Setup complete! You're ready to use Milvus.")
    else:
        print("⚠ Setup partially complete. Review errors above.")
    
    return verification_passed


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


