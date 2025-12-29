"""
LLM service using Google Gemini API for RAG and LLM-as-a-judge
"""
import json
import os
import time
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag.config import GEMINI_API_KEY, GEMINI_MODEL

load_dotenv()

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)


class GeminiLLMService:
    """
    LLM service using Google Gemini API
    """
    
    def __init__(self, model_name: str = GEMINI_MODEL):
        """
        Initialize Gemini LLM service
        
        Args:
            model_name: Gemini model name
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            google_api_key=GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        )
        self.raw_model = genai.GenerativeModel(model_name)
    
    def generate_response(self, prompt: str, context: Optional[str] = None, 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Generate response using Gemini
        
        Args:
            prompt: User prompt/question
            context: Retrieved context documents
            conversation_history: Previous conversation messages
            
        Returns:
            Generated response text
        """
        # Build system message with context
        system_content = """You are a helpful assistant that answers questions about technology products and articles.
Answer based on the provided context. If the context doesn't contain enough information, say so.
Always cite sources when possible."""
        
        messages = [SystemMessage(content=system_content)]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
        
        # Add context if provided
        if context:
            context_prompt = f"Context:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"
            messages.append(HumanMessage(content=context_prompt))
        else:
            messages.append(HumanMessage(content=prompt))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def format_rag_prompt(self, query: str, documents: List[Dict]) -> str:
        """
        Format prompt for RAG with retrieved documents
        
        Args:
            query: User query
            documents: List of document dicts with 'content' and 'metadata'
            
        Returns:
            Formatted prompt string
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('metadata', {}).get('title', 'Unknown')
            url = doc.get('metadata', {}).get('source_url', '')
            content = doc.get('content', '')
            
            context_parts.append(f"[Document {i}]\nTitle: {title}\nURL: {url}\nContent: {content}\n")
        
        context = "\n".join(context_parts)
        
        prompt = f"""Based on the following documents, answer the question. Cite which document(s) you used.

{context}

Question: {query}

Answer:"""
        
        return prompt


class LLMJudgeService:
    """
    LLM-as-a-judge service for evaluation
    """
    
    def __init__(self, model_name: str = GEMINI_MODEL):
        """
        Initialize LLM-as-a-judge service
        
        Args:
            model_name: Gemini model name
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def evaluate_answer_relevance(self, query: str, answer: str, context: str) -> Dict[str, float]:
        """
        Evaluate answer relevance to query
        
        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dictionary with relevance score and explanation
        """
        prompt = f"""You are evaluating an AI assistant's answer. Rate how relevant the answer is to the query.

Query: {query}

Answer: {answer}

Context used: {context}

Rate the relevance on a scale of 1-5 (1=not relevant, 5=highly relevant).
Respond with a JSON object: {{"score": <number>, "explanation": "<brief explanation>"}}"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            return {
                "score": float(result.get("score", 0)),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            print(f"Error in relevance evaluation: {e}")
            return {"score": 0.0, "explanation": f"Error: {str(e)}"}
    
    def evaluate_faithfulness(self, answer: str, context: str) -> Dict[str, float]:
        """
        Evaluate faithfulness of answer to context (no hallucinations)
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dictionary with faithfulness score and explanation
        """
        prompt = f"""You are evaluating an AI assistant's answer. Check if the answer is faithful to the provided context (no hallucinations or made-up information).

Answer: {answer}

Context: {context}

Rate the faithfulness on a scale of 1-5 (1=major hallucinations, 5=completely faithful).
Respond with a JSON object: {{"score": <number>, "explanation": "<brief explanation>"}}"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            return {
                "score": float(result.get("score", 0)),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")
            return {"score": 0.0, "explanation": f"Error: {str(e)}"}
    
    def evaluate_context_utilization(self, answer: str, context: str) -> Dict[str, float]:
        """
        Evaluate how well the answer utilizes the provided context
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dictionary with utilization score and explanation
        """
        prompt = f"""You are evaluating an AI assistant's answer. Check how well the answer utilizes the provided context.

Answer: {answer}

Context: {context}

Rate the context utilization on a scale of 1-5 (1=ignores context, 5=fully utilizes context).
Respond with a JSON object: {{"score": <number>, "explanation": "<brief explanation>"}}"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            return {
                "score": float(result.get("score", 0)),
                "explanation": result.get("explanation", "")
            }
        except Exception as e:
            print(f"Error in context utilization evaluation: {e}")
            return {"score": 0.0, "explanation": f"Error: {str(e)}"}
    
    def evaluate_comprehensive(self, query: str, answer: str, context: str) -> Dict[str, any]:
        """
        Comprehensive evaluation combining all metrics
        
        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Dictionary with all evaluation scores
        """
        relevance = self.evaluate_answer_relevance(query, answer, context)
        faithfulness = self.evaluate_faithfulness(answer, context)
        utilization = self.evaluate_context_utilization(answer, context)
        
        # Add small delay to avoid rate limits
        time.sleep(0.5)
        
        return {
            "relevance": relevance,
            "faithfulness": faithfulness,
            "context_utilization": utilization,
            "average_score": (relevance["score"] + faithfulness["score"] + utilization["score"]) / 3.0
        }