from typing import List, Dict, Any, Optional
import re
from io import BytesIO
import PyPDF2
from docx import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()


class DocumentParser:
    """Parse different document formats"""
    
    @staticmethod
    def parse_txt(file_content: bytes) -> str:
        """Parse text file"""
        return file_content.decode('utf-8', errors='ignore')
    
    @staticmethod
    def parse_pdf(file_content: bytes) -> str:
        """Parse PDF file"""
        pdf_file = BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    
    @staticmethod
    def parse_docx(file_content: bytes) -> str:
        """Parse Word document"""
        doc_file = BytesIO(file_content)
        doc = Document(doc_file)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    
    @staticmethod
    def parse_document(filename: str, file_content: bytes) -> str:
        """Parse document based on file extension"""
        if filename.endswith('.txt'):
            return DocumentParser.parse_txt(file_content)
        elif filename.endswith('.pdf'):
            return DocumentParser.parse_pdf(file_content)
        elif filename.endswith('.docx'):
            return DocumentParser.parse_docx(file_content)
        else:
            raise ValueError(f"Unsupported file type: {filename}")


class TextChunker:
    """Chunk text into manageable pieces"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        return self.splitter.split_text(text)


class VectorStore:
    """Simple vector store for document embeddings"""
    
    def __init__(self, api_key: str = None):
        # Set HuggingFace API token if provided
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        
        # Use HuggingFace embeddings
        self.embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict[str, Any]] = []
    
    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]):
        """Add document chunks to vector store"""
        self.chunks.extend(chunks)
        self.metadata.extend(metadata)
        
        # Generate embeddings
        new_embeddings = self.embeddings_model.embed_documents(chunks)
        new_embeddings_array = np.array(new_embeddings)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings_array
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings_array])
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        
        # Embed query
        query_embedding = self.embeddings_model.embed_query(query)
        query_embedding_array = np.array(query_embedding).reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding_array, self.embeddings)[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx]),
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def clear(self):
        """Clear all stored documents"""
        self.chunks = []
        self.embeddings = None
        self.metadata = []


class RAGPipeline:
    """Complete RAG pipeline"""
    
    def __init__(self, api_key: str):
        self.parser = DocumentParser()
        self.chunker = TextChunker()
        self.vector_store = VectorStore(api_key)
    
    def ingest_document(self, filename: str, file_content: bytes) -> Dict[str, Any]:
        """Ingest a document into the RAG system"""
        try:
            # Parse document
            text = self.parser.parse_document(filename, file_content)
            
            # Chunk text
            chunks = self.chunker.chunk_text(text)
            
            # Create metadata
            metadata = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadata)
            
            return {
                "status": "success",
                "filename": filename,
                "chunks_created": len(chunks),
                "total_chunks": len(self.vector_store.chunks)
            }
        except Exception as e:
            return {
                "status": "error",
                "filename": filename,
                "error": str(e)
            }
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        return self.vector_store.search(query, top_k)
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """Get formatted context for a query"""
        results = self.retrieve(query, top_k)
        
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Document {i} - {result['metadata']['source']}]\n{result['chunk']}"
            )
        
        return "\n\n".join(context_parts)
    
    def clear_documents(self):
        """Clear all documents from the system"""
        self.vector_store.clear()
