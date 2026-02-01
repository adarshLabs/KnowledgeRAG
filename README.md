# QueryX: RAG-Based Chatbot

QueryX is an intelligent chatbot designed to answer questions based on your own documents (PDFs). It uses Retrieval-Augmented Generation (RAG) to provide accurate, cited answers from your knowledge base.

## Features
- **Custom Knowledge Base**: Add your own PDFs to the `docs/` folder.
- **Auto-Sync**: Simply click "Reload Knowledge Base" to index new or modified files.
- **Cited Answers**: Responses include specific source citations (e.g., `[report.pdf]`).
- **Google Gemini Powered**: Uses Gemini Pro for reasoning and Gemini Embeddings for retrieval.
- **Indian Flag Themed UI**: A unique, patriotic interface built with Streamlit.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd RAG_Based_Chatbot
   ```

2. **Install Dependencies**:
   It is recommended to use a virtual environment.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

1. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

2. **Add Documents**:
   - Place your PDF files in the `docs/` folder.
   - Initial loading might take a moment. 
   - If you add files while the app is running, click **"Reload Knowledge Base"** in the sidebar to update the database.

## Tech Stack
- **Framework**: Streamlit
- **LLM**: Google Gemini 1.5 Flash
- **Vector Store**: ChromaDB
- **Orchestration**: LangChain & LangGraph
