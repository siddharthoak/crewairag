import streamlit as st
import os
import sys

# ✅ CRITICAL: Disable CrewAI telemetry BEFORE importing CrewAI
os.environ["OTEL_SDK_DISABLED"] = "true"

from crewrag.src.crew.search_crew import SearchCrew

st.set_page_config(page_title="Search RAG Chat", layout="wide")

st.title("🔍 Search RAG Assistant")
st.caption("Powered by Ollama + Qdrant + CrewAI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize crew once (reuse across queries)
if "crew" not in st.session_state:
    with st.spinner("Initializing AI crew..."):
        try:
            st.session_state.crew = SearchCrew()
            st.success("✅ Crew initialized successfully!")
        except Exception as e:
            st.error(f"Failed to initialize crew: {e}")
            st.stop()

def kickoff_crew(query: str) -> str:
    """Run the SearchCrew with the given query"""
    try:
        # Use the cached crew instance
        result = st.session_state.crew.crew().kickoff(inputs={'query': query})
        return str(result.raw) if hasattr(result, 'raw') else str(result)
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This assistant:
    1. 🔍 Searches PDF knowledge base first
    2. 🌐 Falls back to web search if needed
    3. 📝 Synthesizes information from both sources
    
    **Model:** Ollama llama3.2:1b  
    **Vector DB:** Qdrant  
    **Framework:** CrewAI
    """)
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        # Create a placeholder for the response
        response_placeholder = st.empty()
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Call the crew function
                response = kickoff_crew(prompt)
                response_placeholder.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
            except Exception as e:
                error_msg = f"❌ Error processing query: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })