import streamlit as st
import time
from datetime import datetime
from typing import Dict, Any
from agent import ReasoningEngine, NetworkMonitor
from tools import ToolRouter
from rag_pipeline import RAGPipeline
import os

# Page config
st.set_page_config(page_title="Adaptive Reasoning Agent", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "network_speed" not in st.session_state:
    st.session_state.network_speed = "auto"
if "reasoning_mode" not in st.session_state:
    st.session_state.reasoning_mode = "auto"
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None
if "tool_router" not in st.session_state:
    st.session_state.tool_router = None
if "reasoning_engine" not in st.session_state:
    st.session_state.reasoning_engine = None

# Sidebar for configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Keys
    st.subheader("API Keys")
    hf_key = st.text_input("HuggingFace API Key", type="password", key="hf_key")
    serper_key = st.text_input("Serper API Key (optional)", type="password", key="serper_key")
    
    # Initialize components when API key is provided
    if hf_key and st.session_state.reasoning_engine is None:
        try:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_key
            st.session_state.reasoning_engine = ReasoningEngine(hf_key)
            st.session_state.rag_pipeline = RAGPipeline(hf_key)
            st.session_state.tool_router = ToolRouter(serper_key if serper_key else None)
            st.success("✅ Agent initialized with HuggingFace Mistral-7B!")
        except Exception as e:
            st.error(f"❌ Initialization failed: {str(e)}")
            st.info("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
            # Initialize with None to prevent errors
            if st.session_state.reasoning_engine is None:
                st.session_state.reasoning_engine = None
            if st.session_state.rag_pipeline is None:
                st.session_state.rag_pipeline = None
            if st.session_state.tool_router is None:
                st.session_state.tool_router = None
    
    st.divider()
    
    # Network simulation
    st.subheader("Network Settings")
    network_mode = st.selectbox(
        "Network Mode",
        ["auto", "slow", "medium", "fast"],
        help="Auto mode detects network speed automatically"
    )
    
    if network_mode == "auto":
        if st.button("🔍 Detect Network Speed"):
            with st.spinner("Measuring network..."):
                detected_speed = NetworkMonitor.get_network_speed()
                st.info(f"Detected: {detected_speed}")
                st.session_state.network_speed = detected_speed
    else:
        st.session_state.network_speed = network_mode
    
    st.caption(f"Current: {st.session_state.network_speed}")
    
    # Reasoning mode override
    st.subheader("Reasoning Mode")
    reasoning_override = st.selectbox(
        "Override Reasoning",
        ["auto", "fast", "standard", "deep"],
        help="Auto mode adapts to network conditions"
    )
    st.session_state.reasoning_mode = reasoning_override
    
    st.divider()
    
    # Document upload for RAG
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents for RAG",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx"]
    )
    
    if uploaded_files and st.session_state.rag_pipeline:
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    file_content = uploaded_file.read()
                    result = st.session_state.rag_pipeline.ingest_document(
                        uploaded_file.name,
                        file_content
                    )
                    if result["status"] == "success":
                        st.success(f"✅ {uploaded_file.name}: {result['chunks_created']} chunks")
                    else:
                        st.error(f"❌ {uploaded_file.name}: {result['error']}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🗑️ Clear Documents"):
        if st.session_state.rag_pipeline:
            st.session_state.rag_pipeline.clear_documents()
            st.success("Documents cleared!")

# Main title
st.title("🤖 Adaptive Reasoning Agent")
st.caption("Reasoning depth adapts to network conditions • Answer quality stays constant")

# Display current status
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Network", st.session_state.network_speed.upper())
with col2:
    mode_display = st.session_state.reasoning_mode
    if mode_display == "auto":
        mode_map = {"slow": "fast", "medium": "standard", "fast": "deep"}
        mode_display = f"auto → {mode_map.get(st.session_state.network_speed, 'standard')}"
    st.metric("Reasoning", mode_display.upper())
with col3:
    doc_count = len(st.session_state.rag_pipeline.vector_store.chunks) if st.session_state.rag_pipeline else 0
    st.metric("Documents", f"{doc_count} chunks")

st.divider()

st.divider()

# Tool showcase
with st.expander("🛠️ Available Tools & Examples", expanded=False):
    st.markdown("""
    ### Intelligent Tool Routing
    
    The agent automatically detects and uses these tools:
    
    **1. Web Search** (Shallow/Deep)
    - Try: "Search for latest AI news"
    - Try: "Find information about Python 3.13"
    
    **2. Document Creation** (PDF, Word, Excel)
    - Try: "Create a PDF summary of machine learning"
    - Try: "Generate a Word document about climate change"
    
    **3. DateTime Awareness**
    - Try: "What time is it now?"
    - Try: "What day is today?"
    
    **4. RAG Tool** (Upload documents first)
    - Upload PDFs/DOCX/TXT in sidebar
    - Try: "What does my document say about [topic]?"
    
    **5. Combined Tools**
    - Try: "Search for Python tutorials and create a PDF summary"
    - Try: "What's the current date and search for today's tech news"
    """)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "reasoning" in message and message["reasoning"]:
            with st.expander("🧠 Reasoning Process"):
                st.json(message["reasoning"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    if not hf_key:
        st.error("⚠️ Please enter your HuggingFace API key in the sidebar")
        st.stop()
    
    if not st.session_state.reasoning_engine or not st.session_state.tool_router:
        st.error("⚠️ Agent not initialized. Please check your API key and dependencies.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        reasoning_expander = st.expander("🧠 Reasoning Process", expanded=False)
        
        with st.spinner("Thinking..."):
            # Get network speed
            network_speed = st.session_state.network_speed
            if network_speed == "auto":
                network_speed = NetworkMonitor.get_network_speed()
                st.session_state.network_speed = network_speed
            
            # Detect needed tools
            needed_tools = st.session_state.tool_router.detect_tool_need(prompt)
            
            # Get RAG context if available
            rag_context = ""
            if st.session_state.rag_pipeline and len(st.session_state.rag_pipeline.vector_store.chunks) > 0:
                rag_context = st.session_state.rag_pipeline.get_context(prompt, top_k=3)
            
            # Build context
            context = {
                "rag_context": rag_context,
                "needed_tools": needed_tools,
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Execute tools if needed
            tool_results = {}
            tool_outputs = {}
            for tool in needed_tools:
                if tool == "web_search":
                    depth = "shallow" if network_speed == "slow" else "deep"
                    result = st.session_state.tool_router.route(prompt, tool, depth=depth)
                    tool_results[tool] = result
                    if "results" in result:
                        st.info(f"🔍 Web Search ({depth}): Found {result['count']} results")
                elif tool == "datetime":
                    result = st.session_state.tool_router.route(prompt, tool)
                    tool_results[tool] = result
                    st.info(f"🕐 DateTime: {result.get('datetime', 'N/A')}")
                elif tool == "create_pdf":
                    # Will be created after getting response
                    pass
                elif tool == "create_word":
                    # Will be created after getting response
                    pass
                elif tool == "create_excel":
                    # Will be created after getting response
                    pass
            
            context["tool_results"] = tool_results
            
            # Enhance query with context
            enhanced_query = prompt
            if rag_context:
                enhanced_query = f"""Context from documents:
{rag_context}

User question: {prompt}"""
            
            if tool_results:
                enhanced_query += f"\n\nTool results: {tool_results}"
            
            # Get reasoning mode
            mode_override = st.session_state.reasoning_mode if st.session_state.reasoning_mode != "auto" else None
            
            # Reason and respond
            result = st.session_state.reasoning_engine.reason(
                enhanced_query,
                context,
                network_speed,
                mode_override
            )
            
            # Display response
            response_placeholder.markdown(result["answer"])
            
            # Handle document creation if requested
            if "create_pdf" in needed_tools or "pdf" in prompt.lower():
                pdf_buffer = st.session_state.tool_router.route(
                    prompt, "create_pdf", 
                    content=result["answer"], 
                    filename="response.pdf"
                )
                st.download_button(
                    label="📄 Download as PDF",
                    data=pdf_buffer,
                    file_name="response.pdf",
                    mime="application/pdf"
                )
            
            if "create_word" in needed_tools or "word" in prompt.lower() or "docx" in prompt.lower():
                word_buffer = st.session_state.tool_router.route(
                    prompt, "create_word",
                    content=result["answer"],
                    filename="response.docx"
                )
                st.download_button(
                    label="📝 Download as Word",
                    data=word_buffer,
                    file_name="response.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            
            # Display reasoning
            with reasoning_expander:
                st.json({
                    "mode": result["mode"],
                    "network_speed": result["network_speed"],
                    "steps": result["steps"],
                    "execution_time": f"{result['execution_time']:.2f}s",
                    "tools_detected": needed_tools,
                    "tools_used": result.get("tools_used", []),
                    "rag_chunks_used": len(rag_context) > 0
                })
                
                if "reasoning" in result:
                    st.subheader("Reasoning Steps")
                    for step in result["reasoning"]:
                        if isinstance(step, dict):
                            st.write(f"**{step['step'].title()}:**")
                            st.write(step['content'][:200] + "..." if len(step['content']) > 200 else step['content'])
                        else:
                            st.write(step)
            
            # Add to message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "reasoning": {
                    "mode": result["mode"],
                    "network_speed": result["network_speed"],
                    "execution_time": f"{result['execution_time']:.2f}s"
                }
            })
