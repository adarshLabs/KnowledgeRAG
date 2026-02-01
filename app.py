import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from sync_embeddings import process_documents

from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.tools import tool
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="QueryX", page_icon="🇮🇳", layout="wide")

# --- Custom CSS for Indian Flag Theme ---
st.markdown("""
    <style>
    /* Gradient Background for the Header/Top Line */
    .stApp > header {
        background: linear-gradient(to right, #FF9933, #FFFFFF, #138808) !important;
    }
    
    /* Main Title Styling */
    h1 {
        color: #000080; /* Navy Blue for Chakra color */
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
    
    /* User Message - Saffron Tint */
    div[data-testid="stChatMessage"] :nth-child(1) {
       /* background-color: #FFF0E0; */
    }

    /* Assistant Message - Green Tint */
    div[data-testid="stChatMessage"] :nth-child(2) {
       /* background-color: #E0F0E0; */
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(to right, #FF9933, #138808);
        color: white;
        border: none;
        border-radius: 20px;
    }
    
    /* Borders */
    .stChatInput {
        border: 2px solid #FF9933 !important;
        border-radius: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Banner ---
st.markdown("""
<div style="background: linear-gradient(90deg, #FF9933 33%, #FFFFFF 33%, #FFFFFF 66%, #138808 66%); height: 10px; width: 100%; border-radius: 5px; margin-bottom: 20px;"></div>
""", unsafe_allow_html=True)

st.title("QueryX: RAG based Chatbot")

# --- RAG Agent Initialization (Cached) ---
@st.cache_resource
def load_models():
    """Load and cache the heavy AI models."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return llm, embeddings

def get_agent(llm, embeddings):
    """Create the agent graph with a fresh database connection."""
    persist_directory = "chroma_db"
    collection_name = "Embeddings"
    
    if not os.path.exists(persist_directory):
        st.error("Database not found. Please run the sync script first.")
        st.stop()
    
    # Initialize Chroma without caching to ensure we see latest updates
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    @tool
    def retriever_tool(query: str) -> str:
        """Searches and returns information from the available documents."""
        docs = retriever.invoke(query)
        if not docs:
            return "I found no relevant information in the document."
        formatted_docs = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source)
            formatted_docs.append(f"Source: {filename}\nContent:\n{doc.page_content}")
        return "\n\n".join(formatted_docs)

    tools = [retriever_tool]
    tools_dict = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    
    class AgentState(TypedDict):
        messages: Annotated[Sequence[HumanMessage | AIMessage | ToolMessage], add_messages]

    def should_continue(state: AgentState):
        last_message = state['messages'][-1]
        return hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0

    system_prompt = """
        You are QueryX, an intelligent Retrieval-Augmented Generation (RAG) assistant.

        Your role is to answer user questions strictly using the information retrieved from the provided documents via the retriever tool.

        Rules you must follow:
        1. Always use the retriever tool before answering any question.
        2. Base your answers only on the retrieved content. Do not use prior knowledge or make assumptions.
        3. If the retrieved documents do not contain sufficient information, clearly say:
        "The provided documents do not contain enough information to answer this question."
        4. Cite specific document sources for every factual statement.
        5. Use concise, clear, and professional language.
        6. Do not hallucinate facts, examples, or explanations.
        7. If multiple documents are relevant, synthesize the answer while citing all applicable sources.

        Citation format:
        - Cite sources inline using: [Document Name, Section/Page]
        - Example: “This process improves retrieval accuracy [Architecture.pdf, Section 3.2].”

        Your goal is to provide accurate, trustworthy, and verifiable answers grounded entirely in the retrieved documents.
    """

    def call_llm(state: AgentState):
        messages = list(state['messages'])
        messages = [SystemMessage(content=system_prompt)] + messages 
        response = llm_with_tools.invoke(messages)
        return {'messages': [response]}

    def take_action(state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            if t['name'] in tools_dict:
                res = tools_dict[t['name']].invoke(t['args'].get('query', ''))
                results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(res)))
        return {'messages': results}

    graph = StateGraph(AgentState)
    graph.add_node("llm", call_llm)
    graph.add_node("retriever_agent", take_action)
    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
    graph.add_edge("retriever_agent", "llm")
    graph.set_entry_point("llm")
    
    return graph.compile()

# Initialize the agent
try:
    llm, embeddings = load_models()
    rag_agent = get_agent(llm, embeddings)
except Exception as e:
    st.error(f"Error initializing RAG Agent: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    if st.button("Reload Knowledge Base"):
        # Trigger the synchronization process to detect new files
        with st.spinner("Syncing documents..."):
            process_documents()
        
        # We don't strictly need to clear cache_resource if we separated the DB loading,
        # but clearing it ensures we also get fresh models if needed (rare).
        # Simply rerunning will now force get_agent() to run again and reconnect to DB.
        st.cache_resource.clear() 
        st.success("Refreshed!")
        st.rerun()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing database..."):
            try:
                # Convert session history to LangChain format for the agent context if desired, 
                # but simplified agent just takes the new human message usually for this stateless graph 
                # or we pass the whole history if we want multi-turn memory. 
                # For this specific graph structure, we likely just pass the new message.
                
                # To support conversation history, we'd need to reconstruct the message objects.
                # For now, let's keep it simple and just send the latest prompt as per the original script style
                # or improved: convert session state to proper objects? 
                
                # Let's simple pass the user query. The graph itself handles the flow.
                # If we want memory, we should pass history.
                
                messages_for_agent = [HumanMessage(content=prompt)]
                result = rag_agent.invoke({"messages": messages_for_agent})
                
                # Extract final response
                final_content = result['messages'][-1].content
                
                # Handle structured output if any (reusing fix from before)
                if isinstance(final_content, list):
                    text_parts = []
                    for block in final_content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text_parts.append(block.get('text'))
                    response_text = "\n".join(text_parts)
                else:
                    response_text = str(final_content)
                
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
