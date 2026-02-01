import os
import hashlib
import json
import shutil
from typing import Dict, List
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
DOCS_DIR = "docs"
PERSIST_DIRECTORY = "chroma_db"
COLLECTION_NAME = "Embeddings"
STATE_FILE = "processed_state.json"

def get_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_state() -> Dict[str, str]:
    """Loads the processing state from a JSON file."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_state(state: Dict[str, str]):
    """Saves the processing state to a JSON file."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=4)

def initialize_vectorstore():
    """Initializes and returns the Chroma vector store."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

def process_documents():
    """Syncs documents in the docs folder with the ChromaDB vector store."""
    if not os.path.exists(DOCS_DIR):
        print(f"Creating {DOCS_DIR} directory...")
        os.makedirs(DOCS_DIR)

    state = load_state()
    current_files = {}
    
    # scan current files
    for filename in os.listdir(DOCS_DIR):
        if filename.endswith(".pdf"):
            filepath = os.path.join(DOCS_DIR, filename)
            current_files[filename] = get_file_hash(filepath)
    
    vectorstore = initialize_vectorstore()
    
    # Identify changes
    files_to_add = [] # List of (filename, filepath)
    files_to_remove = [] # List of filename
    
    # Check for new or modified files
    for filename, filehash in current_files.items():
        if filename not in state:
            print(f"Found new file: {filename}")
            files_to_add.append((filename, os.path.join(DOCS_DIR, filename)))
        elif state[filename] != filehash:
            print(f"File modified: {filename}")
            # storage for modified file: remove then add
            files_to_remove.append(filename)
            files_to_add.append((filename, os.path.join(DOCS_DIR, filename)))
            
    # Check for deleted files
    for filename in state:
        if filename not in current_files:
            print(f"File deleted: {filename}")
            files_to_remove.append(filename)

    if not files_to_add and not files_to_remove:
        print("No changes detected.")
        return

    # Handle removals
    if files_to_remove:
        print(f"Removing {len(files_to_remove)} files from database...")
        # To delete from Chroma, we need the IDs. 
        # Since we didn't track IDs explicitly mapped to files in state (simple version),
        # we can query by metadata 'source'. 
        # But standard PyPDFLoader adds 'source' metadata with absolute path.
        # Let's hope the absolute path hasn't changed if we are on same machine.
        
        # Better approach: Fetch all IDs, check metadata.
        # This can be slow for large DBs but fine for this scale.
        
        # Actually, let's just use the `get` method with where filter if possible, 
        # or just iterate. Chroma `get` specific to collection.
        
        all_data = vectorstore.get()
        ids_to_delete = []
        
        for idx, metadata in enumerate(all_data['metadatas']):
            # Metadata source is usually the path
            source_path = metadata.get('source', '')
            # Check if any removed filename is in the source path
            for filename in files_to_remove:
                if filename in source_path:
                    ids_to_delete.append(all_data['ids'][idx])
        
        if ids_to_delete:
            vectorstore.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} chunks.")
        
        for filename in files_to_remove:
            if filename in state:
                del state[filename]

    # Handle additions
    if files_to_add:
        print(f"Adding {len(files_to_add)} files to database...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        files_processed_count = 0
        for filename, filepath in files_to_add:
            try:
                print(f"Processing {filename}...")
                
                # Try PyPDFLoader first
                loader = PyPDFLoader(filepath)
                pages = loader.load()
                
                # Check for empty content
                has_text = any(p.page_content and p.page_content.strip() for p in pages)
                
                if not has_text:
                    print(f"  PyPDFLoader found no text in {filename}. Trying pdfplumber...")
                    try:
                        from langchain_community.document_loaders import PDFPlumberLoader
                        loader = PDFPlumberLoader(filepath)
                        pages = loader.load()
                        has_text = any(p.page_content and p.page_content.strip() for p in pages)
                        if has_text:
                            print("  pdfplumber successfully extracted text.")
                        else:
                            print("  pdfplumber also found no text.")
                    except ImportError:
                        print("  pdfplumber not installed. Install it with `pip install pdfplumber` for better extraction.")
                    except Exception as e:
                        print(f"  pdfplumber failed: {e}")

                chunks = text_splitter.split_documents(pages)
                
                if chunks and has_text:
                    vectorstore.add_documents(chunks)
                    state[filename] = current_files[filename]
                    files_processed_count += 1
                    print(f"Successfully added {filename}")
                else:
                    print(f"Warning: No extractable text found in {filename}. It might be a scanned image.")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    save_state(state)
    print("Sync complete.")

if __name__ == "__main__":
    process_documents()
