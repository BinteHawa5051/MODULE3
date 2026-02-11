ADAPTIVE REASONING AGENT
========================

A Streamlit chatbot that adapts reasoning depth based on network conditions.

FEATURES:
- Adaptive reasoning: Fast, Standard, and Deep modes
- Network-aware: Automatically adjusts to connection speed
- Tool routing: Web search, document creation, datetime
- RAG pipeline: Upload and query your documents
- Real-time streaming responses

SETUP:
1. Activate virtual environment:
   venv\Scripts\activate

2. Install dependencies:
   pip install -r requirements.txt

3. Run the application:
   streamlit run app.py

4. Enter your OpenAI API key in the sidebar
   - Optional: Add Serper API key for web search

USAGE:
- Upload documents (PDF, DOCX, TXT) for RAG
- Ask questions - the agent adapts reasoning to network speed
- View reasoning process in expandable sections
- Switch between auto/manual network and reasoning modes

ARCHITECTURE:
- agent.py: Custom reasoning engine with 3 modes
- tools.py: Web search, document creation, datetime tools
- rag_pipeline.py: Document parsing, chunking, vector search
- app.py: Streamlit UI and orchestration

REASONING MODES:
- Fast: Single-pass (for slow networks)
- Standard: 2-step analysis (for medium networks)
- Deep: 3-step decomposition (for fast networks)

Answer quality remains constant across all modes.
