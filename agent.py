from typing import List, Dict, Any, Literal, Optional
import time
import requests
from datetime import datetime
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import os
import json

load_dotenv()

ReasoningMode = Literal["fast", "standard", "deep"]
NetworkSpeed = Literal["slow", "medium", "fast"]


class NetworkMonitor:
    """Monitors network conditions to determine reasoning strategy"""
    
    @staticmethod
    def measure_latency(test_url: str = "https://www.google.com") -> float:
        """Measure network latency in milliseconds"""
        try:
            start = time.time()
            requests.head(test_url, timeout=5)
            latency = (time.time() - start) * 1000
            return latency
        except:
            return 5000  # Assume slow if test fails
    
    @staticmethod
    def classify_network(latency: float) -> NetworkSpeed:
        """Classify network speed based on latency"""
        if latency < 100:
            return "fast"
        elif latency < 300:
            return "medium"
        else:
            return "slow"
    
    @staticmethod
    def get_network_speed() -> NetworkSpeed:
        """Get current network speed classification"""
        latency = NetworkMonitor.measure_latency()
        return NetworkMonitor.classify_network(latency)


class ReasoningEngine:
    """Custom reasoning engine with adaptive depth"""
    
    def __init__(self, api_key: str = None):
        # Set HuggingFace API token
        if api_key:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        
        # Initialize HuggingFace model
        llm = HuggingFaceEndpoint(
            repo_id='mistralai/Mistral-7B-Instruct-v0.2',
            task='text-generation',
            max_new_tokens=512,
            temperature=0.7,
            top_k=50
        )
        self.llm = ChatHuggingFace(llm=llm)
    
    def select_reasoning_mode(
        self, 
        network_speed: NetworkSpeed,
        override: Optional[ReasoningMode] = None
    ) -> ReasoningMode:
        """Select reasoning mode based on network conditions"""
        if override and override != "auto":
            return override
        
        mode_map = {
            "slow": "fast",
            "medium": "standard",
            "fast": "deep"
        }
        return mode_map[network_speed]
    
    def fast_reasoning(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Single-pass reasoning for fast response"""
        # Build context string
        context_str = f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if context.get("rag_context"):
            context_str += f"\nDocument Context:\n{context['rag_context']}\n"
        
        if context.get("tool_results"):
            context_str += f"\nTool Results:\n{json.dumps(context['tool_results'], indent=2)}\n"
        
        system_prompt = f"""You are a helpful AI assistant. Provide direct, concise answers.
Use the available context and tool results to answer accurately.

{context_str}"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]
        
        response = self.llm.invoke(messages)
        
        return {
            "mode": "fast",
            "steps": 1,
            "reasoning": ["Direct single-pass response with context"],
            "answer": response.content,
            "tools_used": list(context.get("tool_results", {}).keys())
        }
    
    def standard_reasoning(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Step-based reasoning with intermediate analysis"""
        steps = []
        
        # Build context string
        context_str = f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if context.get("rag_context"):
            context_str += f"\nDocument Context:\n{context['rag_context']}\n"
        
        if context.get("tool_results"):
            context_str += f"\nTool Results:\n{json.dumps(context['tool_results'], indent=2)}\n"
        
        # Step 1: Analyze query
        analyze_prompt = f"""Analyze this query and identify:
1. What information is needed
2. What context/tools are available
3. Key aspects to address

{context_str}

Query: {query}

Provide a brief analysis."""
        
        messages = [SystemMessage(content="You are an analytical assistant."), 
                   HumanMessage(content=analyze_prompt)]
        analysis = self.llm.invoke(messages)
        steps.append({"step": "analysis", "content": analysis.content})
        
        # Step 2: Generate response
        response_prompt = f"""Based on this analysis:
{analysis.content}

{context_str}

Answer the original query: {query}

Provide a comprehensive answer using all available context."""
        
        messages = [SystemMessage(content="You are a helpful assistant."),
                   HumanMessage(content=response_prompt)]
        response = self.llm.invoke(messages)
        steps.append({"step": "response", "content": response.content})
        
        return {
            "mode": "standard",
            "steps": 2,
            "reasoning": steps,
            "answer": response.content,
            "tools_used": list(context.get("tool_results", {}).keys())
        }
    
    def deep_reasoning(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Multi-step deep analysis"""
        steps = []
        
        # Build context string
        context_str = f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if context.get("rag_context"):
            context_str += f"\nDocument Context:\n{context['rag_context']}\n"
        
        if context.get("tool_results"):
            context_str += f"\nTool Results:\n{json.dumps(context['tool_results'], indent=2)}\n"
        
        # Step 1: Decompose query
        decompose_prompt = f"""Break down this query into sub-questions:

{context_str}

Query: {query}

List 2-3 sub-questions that need to be answered."""
        
        messages = [SystemMessage(content="You are an analytical assistant."),
                   HumanMessage(content=decompose_prompt)]
        decomposition = self.llm.invoke(messages)
        steps.append({"step": "decomposition", "content": decomposition.content})
        
        # Step 2: Analyze each aspect
        analyze_prompt = f"""For these sub-questions:
{decomposition.content}

{context_str}

Provide detailed analysis for each using all available context."""
        
        messages = [SystemMessage(content="You are a thorough analyst."),
                   HumanMessage(content=analyze_prompt)]
        analysis = self.llm.invoke(messages)
        steps.append({"step": "analysis", "content": analysis.content})
        
        # Step 3: Synthesize answer
        synthesis_prompt = f"""Based on this analysis:
{analysis.content}

{context_str}

Synthesize a comprehensive answer to: {query}"""
        
        messages = [SystemMessage(content="You are a synthesis expert."),
                   HumanMessage(content=synthesis_prompt)]
        response = self.llm.invoke(messages)
        steps.append({"step": "synthesis", "content": response.content})
        
        return {
            "mode": "deep",
            "steps": 3,
            "reasoning": steps,
            "answer": response.content,
            "tools_used": list(context.get("tool_results", {}).keys())
        }
    
    def reason(
        self,
        query: str,
        context: Dict[str, Any],
        network_speed: NetworkSpeed,
        mode_override: Optional[ReasoningMode] = None
    ) -> Dict[str, Any]:
        """Main reasoning method that adapts to network conditions"""
        mode = self.select_reasoning_mode(network_speed, mode_override)
        
        start_time = time.time()
        
        if mode == "fast":
            result = self.fast_reasoning(query, context)
        elif mode == "standard":
            result = self.standard_reasoning(query, context)
        else:  # deep
            result = self.deep_reasoning(query, context)
        
        result["execution_time"] = time.time() - start_time
        result["network_speed"] = network_speed
        
        return result
