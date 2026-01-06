"""
LLM service using Google Gemini API for RAG and LLM-as-a-judge
"""
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from rag.config import GEMINI_API_KEY, GEMINI_MODEL
from rag.prompts import QUERY_DECOMPOSITION_PROMPT, HYDE_PROMPT, RAG_SYSTEM_PROMPT

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
            google_api_key=GEMINI_API_KEY or os.getenv("GEMINI_API_KEY"),
            convert_system_message_to_human=True,
        )
        self.raw_model = genai.GenerativeModel(model_name)
    
    def generate_search_queries(self, query: str) -> List[str]:
        """
        Decompose a complex query into simple search queries
        
        Args:
            query: User's original query
            
        Returns:
            List of simplified search queries
        """
        try:
            prompt = QUERY_DECOMPOSITION_PROMPT.format(query=query)
            response = self.raw_model.generate_content(
                prompt,
                generation_config={"response_mime_type": "application/json"}
            )
            result = json.loads(response.text)
            queries = result.get("queries", [])
            
            # Ensure the original query is included if the list is empty or doesn't have it
            if not queries:
                queries = [query]
            elif query not in queries:
                queries.append(query)
                
            return queries
        except Exception as e:
            print(f"Error generating search queries: {e}")
            return [query]

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer (HyDE) for the query
        
        Args:
            query: User's query
            
        Returns:
            Hypothetical answer text
        """
        try:
            prompt = HYDE_PROMPT.format(query=query)
            response = self.raw_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating HyDE answer: {e}")
            return ""

    def generate_response(self, prompt: str, context: Optional[str] = None, 
                         conversation_history: Optional[List[Dict[str, str]]] = None,
                         current_date: Optional[str] = None) -> str:
        """
        Generate response using Gemini with Chain-of-Thought reasoning
        
        Args:
            prompt: User prompt/question
            context: Retrieved context documents
            conversation_history: Previous conversation messages
            current_date: Current date string (YYYY-MM-DD)
            
        Returns:
            Generated response text
        """
        # Default current date if not provided
        if not current_date:
            current_date = datetime.now().strftime("%Y-%m-%d")

        # Build system message with context
        if context:
            system_content = RAG_SYSTEM_PROMPT.format(context=context, current_date=current_date)
        else:
            system_content = f"You are a helpful assistant. Current Date: {current_date}. Please answer the user's question."
        
        messages = [SystemMessage(content=system_content)]
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
        
        # Add current user prompt
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
