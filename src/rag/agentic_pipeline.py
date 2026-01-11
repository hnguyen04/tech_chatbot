"""
AgenticRAG Pipeline with Query Decomposition strategy.
Uses the same decomposition approach as UnifiedRAGPipeline for reliable retrieval.
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Set
from langchain_core.documents import Document

from rag.agent_tools import TOOL_DEFINITIONS, ToolExecutor, format_tools_for_prompt
from rag.prompts import RAG_SYSTEM_PROMPT, AGENT_PLANNING_PROMPT, AGENT_REFLECTION_PROMPT
from rag.llm_service import GeminiLLMService, OpenAILLMService
from rag.vietnamese_retriever import VietnameseRetriever
from rag.utils import reciprocal_rank_fusion
from storage.postgres_client import PostgresClient
from rag.config import MILVUS_COLLECTION_NAME, TOP_K_RETRIEVE, TOP_N_RERANK


class AgenticRAGPipeline:
    """
    ReAct-style agentic RAG pipeline.
    
    Flow: Plan → Act → Observe → Reflect → (repeat or Answer)
    """
    
    MAX_ITERATIONS = 4  # Prevent infinite loops
    
    def __init__(
        self,
        llm_service: Optional[Union[GeminiLLMService, OpenAILLMService]] = None,
        retriever: Optional[VietnameseRetriever] = None,
        pg_client: Optional[PostgresClient] = None,
        collection_name: Optional[str] = None,
        use_openai: bool = True
    ):
        """
        Initialize agentic pipeline.
        
        Args:
            llm_service: LLM service for generation (Gemini or OpenAI)
            retriever: Vector retriever
            pg_client: PostgreSQL client
            collection_name: Milvus collection name
            use_openai: If True and no llm_service provided, use OpenAI (default for Deep mode)
        """
        print("Initializing AgenticRAG Pipeline...")
        
        # Default to OpenAI for Deep mode, fall back to Gemini if OpenAI not available
        if llm_service:
            self.llm_service = llm_service
        elif use_openai:
            try:
                self.llm_service = OpenAILLMService()
            except ValueError as e:
                print(f"OpenAI not available ({e}), falling back to Gemini")
                self.llm_service = GeminiLLMService()
        else:
            self.llm_service = GeminiLLMService()
        
        self.is_openai = isinstance(self.llm_service, OpenAILLMService)
        
        target_collection = collection_name or MILVUS_COLLECTION_NAME
        self.retriever = retriever or VietnameseRetriever(collection_name=target_collection)
        self.pg_client = pg_client or PostgresClient()
        
        # Initialize tool executor with dependencies
        self.tool_executor = ToolExecutor(
            retriever=self.retriever,
            pg_client=self.pg_client
        )
        
        self.conversation_history: List[Dict[str, str]] = []
        
        print("AgenticRAG Pipeline initialized")
    
    def query(self, user_query: str, use_history: bool = True) -> Dict[str, Any]:
        """
        Process a query using agentic reasoning.
        
        Args:
            user_query: User's question
            use_history: Whether to use conversation history
            
        Returns:
            Response dict with answer, sources, and metadata
        """
        print(f"\n[AgenticRAG] Processing: {user_query}")
        
        # Track gathered context and actions
        context_items: List[Dict[str, Any]] = []
        action_log: List[str] = []
        
        for iteration in range(self.MAX_ITERATIONS):
            print(f"\n[Iteration {iteration + 1}/{self.MAX_ITERATIONS}]")
            
            # PLAN: Decide next action
            action = self._plan_next_action(user_query, context_items)
            action_log.append(f"Iteration {iteration + 1}: {action}")
            
            if action.get("action") == "ANSWER":
                print("  → Ready to generate answer")
                break
            
            # Handle both formats:
            # 1. {"action": "TOOL", "tool": "tool_name", "params": {...}}
            # 2. {"action": "tool_name", "params": {...}} (LLM sometimes does this)
            action_type = action.get("action", "")
            tool_name = action.get("tool", "")
            params = action.get("params", {})
            
            # Check if action is "TOOL" or if action itself is a valid tool name
            valid_tools = list(self.tool_executor._tools.keys())
            is_tool_action = action_type == "TOOL" or action_type in valid_tools
            
            if is_tool_action:
                # If action is the tool name directly, use it
                if action_type in valid_tools:
                    tool_name = action_type
                
                print(f"  → Executing tool: {tool_name}({params})")
                
                # ACT: Execute the tool
                result = self.tool_executor.execute(tool_name, params)
                
                # Log result count for debugging
                if result.get("success"):
                    data = result.get("data", [])
                    count = len(data) if isinstance(data, list) else (1 if data else 0)
                    print(f"  → Tool returned {count} result(s)")
                else:
                    print(f"  → Tool error: {result.get('error', 'unknown')}")
                
                # Store result in context
                context_items.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result
                })
                
                # REFLECT: Check if we have enough info
                if self._has_enough_info(user_query, context_items):
                    print("  → Sufficient information gathered")
                    break
            else:
                print(f"  → Unknown action: {action}")
                break
        
        # GENERATE: Create final answer
        print("\n[Generating final answer...]")
        answer, sources = self._generate_answer(user_query, context_items, use_history)
        
        return {
            "answer": answer,
            "sources": sources,
            "iterations": len(action_log),
            "action_log": action_log,
            "query": user_query
        }
    
    def _plan_next_action(self, query: str, context: List[Dict]) -> Dict[str, Any]:
        """
        Use LLM to decide the next action.
        
        Args:
            query: User query
            context: Gathered context so far
            
        Returns:
            Action dict with 'action', 'tool', 'params'
        """
        # Format context for prompt
        context_str = self._format_context_for_planning(context)
        tools_str = format_tools_for_prompt()
        
        prompt = AGENT_PLANNING_PROMPT.format(
            tools=tools_str,
            context=context_str if context_str else "No information gathered yet.",
            query=query
        )
        
        try:
            # Use appropriate method based on LLM type
            if self.is_openai:
                action = self.llm_service.generate_json_response(prompt)
            else:
                # Gemini path
                response = self.llm_service.raw_model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                action = json.loads(response.text)
            return action if action else {"action": "ANSWER"}
        except Exception as e:
            print(f"Error in planning: {e}")
            # Fallback to answering with what we have
            return {"action": "ANSWER"}
    
    def _has_enough_info(self, query: str, context: List[Dict]) -> bool:
        """
        Use LLM to determine if we have enough information.
        
        Args:
            query: User query
            context: Gathered context
            
        Returns:
            True if sufficient info gathered
        """
        # Simple heuristic: if we have at least 2 context items, check with LLM
        if len(context) < 1:
            return False
        
        # Quick heuristic for comparison queries: if we have 2+ successful tool calls
        # with results, we likely have enough info
        successful_tools = sum(1 for c in context if c.get("result", {}).get("success"))
        if successful_tools >= 2:
            # Check if we have results for different queries (not duplicates)
            unique_queries = set()
            for c in context:
                params = c.get("params", {})
                query_param = params.get("query", "")
                if query_param:
                    unique_queries.add(query_param.lower())
            
            # If we have 2+ unique successful queries, we likely have enough for comparisons
            if len(unique_queries) >= 2:
                print("  → Heuristic: 2+ unique successful searches, marking as sufficient")
                return True
        
        context_str = self._format_context_for_planning(context)
        
        prompt = AGENT_REFLECTION_PROMPT.format(
            query=query,
            context=context_str
        )
        
        try:
            # Use appropriate method based on LLM type
            if self.is_openai:
                result = self.llm_service.generate_json_response(prompt)
            else:
                # Gemini path
                response = self.llm_service.raw_model.generate_content(
                    prompt,
                    generation_config={"response_mime_type": "application/json"}
                )
                result = json.loads(response.text)
            return result.get("enough", False) if result else False
        except Exception as e:
            print(f"Error in reflection: {e}")
            # After 2+ iterations, assume we have enough
            return len(context) >= 2
    
    def _format_context_for_planning(self, context: List[Dict]) -> str:
        """Format gathered context for planning prompt."""
        if not context:
            return ""
        
        parts = []
        for i, item in enumerate(context, 1):
            tool = item.get("tool", "unknown")
            params = item.get("params", {})
            result = item.get("result", {})
            
            # Truncate result data for prompt
            if "data" in result:
                data = result["data"]
                if isinstance(data, list):
                    data_preview = data[:3]  # First 3 items
                    data_str = json.dumps(data_preview, ensure_ascii=False, indent=2)[:1000]
                elif isinstance(data, dict):
                    data_str = json.dumps(data, ensure_ascii=False, indent=2)[:1000]
                else:
                    data_str = str(data)[:1000]
            else:
                data_str = str(result)[:500]
            
            parts.append(f"[Action {i}] {tool}({params})\nResult: {data_str}")
        
        return "\n\n".join(parts)
    
    def _generate_answer(
        self,
        query: str,
        context: List[Dict],
        use_history: bool
    ) -> tuple[str, List[Dict]]:
        """
        Generate final answer from gathered context.
        
        Args:
            query: User query
            context: Gathered context from tools
            use_history: Whether to use conversation history
            
        Returns:
            Tuple of (answer_text, sources_list)
        """
        # Build context from tool results - take TOP items from EACH tool call
        # to ensure we have representation from all searches (important for comparisons)
        documents = []
        sources = []
        seen_ids = set()  # Deduplicate by product ID
        
        # Process each tool call separately to ensure balanced representation
        docs_per_tool = max(5, 15 // max(len(context), 1))  # At least 5 per tool, up to 15 total
        
        for item in context:
            result = item.get("result", {})
            data = result.get("data", [])
            tool_docs_added = 0
            
            entries_to_process = data if isinstance(data, list) else ([data] if isinstance(data, dict) else [])
            
            for entry in entries_to_process:
                if not isinstance(entry, dict):
                    continue
                    
                # Skip duplicates (same product from repeated searches)
                entry_id = entry.get("record_id") or entry.get("id")
                if entry_id and entry_id in seen_ids:
                    continue
                if entry_id:
                    seen_ids.add(entry_id)
                
                # Limit docs per tool call for balanced representation
                if tool_docs_added >= docs_per_tool:
                    break
                
                # Build document content - INCLUDE SPECS if available
                parts = []
                
                # Product identity
                if entry.get("full_title"):
                    parts.append(f"Sản phẩm: {entry['full_title']}")
                if entry.get("brand"):
                    parts.append(f"Thương hiệu: {entry['brand']}")
                if entry.get("model"):
                    parts.append(f"Model: {entry['model']}")
                if entry.get("price"):
                    parts.append(f"Giá: {entry['price']} VNĐ")
                
                # CRITICAL: Include specifications if available
                if entry.get("specifications"):
                    parts.append(f"\nThông số kỹ thuật:\n{entry['specifications']}")
                elif entry.get("content") or entry.get("content_text"):
                    parts.append(f"\nMô tả: {entry.get('content') or entry.get('content_text')}")
                
                doc_content = "\n".join(parts) if parts else json.dumps(entry, ensure_ascii=False)
                documents.append(doc_content[:3000])  # Allow more chars for specs
                tool_docs_added += 1
                
                # Build source info
                sources.append({
                    "record_id": entry_id,
                    "title": entry.get("full_title", ""),
                    "brand": entry.get("brand", ""),
                    "model": entry.get("model", ""),
                    "price": entry.get("price", ""),
                    "url": entry.get("source_url", ""),
                })
        
        # Format context for LLM - use all collected docs (already limited per-tool)
        context_parts = []
        for i, doc in enumerate(documents[:15], 1):  # Increased limit to 15
            context_parts.append(f"[Nguồn {i}]\n{doc}")
        
        context_str = "\n\n".join(context_parts)
        
        # Debug: show how many unique docs are being passed
        print(f"  → Passing {len(context_parts)} unique documents to LLM for answer generation")
        
        # Generate response
        current_date = datetime.now().strftime("%Y-%m-%d")
        history = self.conversation_history if use_history else None
        
        answer = self.llm_service.generate_response(
            prompt=query,
            context=context_str,
            conversation_history=history,
            current_date=current_date
        )
        
        # Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": query})
            self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Dedupe sources
        seen_ids = set()
        unique_sources = []
        for s in sources:
            rid = s.get("record_id")
            if rid and rid not in seen_ids:
                seen_ids.add(rid)
                unique_sources.append(s)
        
        return answer, unique_sources[:5]  # Limit sources
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def query_for_eval(self, user_query: str) -> Dict[str, Any]:
        """
        Process a query and return additional data needed for evaluation.
        
        This method returns more detailed information than query() for
        evaluation purposes, including tool usage and retrieved IDs.
        
        Args:
            user_query: User's question
            
        Returns:
            Dictionary with answer, sources, tool_log, retrieved_ids, etc.
        """
        print(f"\n[AgenticRAG Eval] Processing: {user_query}")
        
        # Track gathered context and actions
        context_items: List[Dict[str, Any]] = []
        action_log: List[str] = []
        tools_used: List[str] = []
        all_retrieved_ids: List[int] = []
        
        for iteration in range(self.MAX_ITERATIONS):
            # PLAN: Decide next action
            action = self._plan_next_action(user_query, context_items)
            action_log.append(f"Iteration {iteration + 1}: {action}")
            
            if action.get("action") == "ANSWER":
                break
            
            # Handle both formats (same as query method)
            action_type = action.get("action", "")
            tool_name = action.get("tool", "")
            params = action.get("params", {})
            
            valid_tools = list(self.tool_executor._tools.keys())
            is_tool_action = action_type == "TOOL" or action_type in valid_tools
            
            if is_tool_action:
                if action_type in valid_tools:
                    tool_name = action_type
                    
                tools_used.append(tool_name)
                
                # ACT: Execute the tool
                result = self.tool_executor.execute(tool_name, params)
                
                # Extract retrieved IDs if applicable
                if result.get("success") and result.get("data"):
                    data = result["data"]
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                rid = item.get("record_id") or item.get("id")
                                if rid:
                                    all_retrieved_ids.append(int(rid))
                    elif isinstance(data, dict):
                        rid = data.get("record_id") or data.get("id")
                        if rid:
                            all_retrieved_ids.append(int(rid))
                
                # Store result in context
                context_items.append({
                    "tool": tool_name,
                    "params": params,
                    "result": result
                })
                
                # REFLECT: Check if we have enough info
                if self._has_enough_info(user_query, context_items):
                    break
            else:
                break
        
        # GENERATE: Create final answer
        answer, sources = self._generate_answer(user_query, context_items, use_history=False)
        
        # Build context string for evaluation
        context_parts = []
        for item in context_items:
            result = item.get("result", {})
            data = result.get("data", [])
            if isinstance(data, list):
                for entry in data[:3]:  # First 3 items
                    if isinstance(entry, dict):
                        content = entry.get("content") or entry.get("content_text", "")
                        if content:
                            context_parts.append(content[:500])
            elif isinstance(data, dict):
                content = data.get("content_text", "")
                if content:
                    context_parts.append(content[:500])
        
        context_str = "\n\n".join(context_parts)
        
        # Extract source IDs
        source_ids = []
        for s in sources:
            rid = s.get("record_id")
            if rid:
                source_ids.append(int(rid) if isinstance(rid, str) else rid)
        
        return {
            "answer": answer,
            "sources": sources,
            "retrieved_ids": all_retrieved_ids,
            "reranked_ids": source_ids,  # Final sources used
            "context": context_str,
            "tool_log": tools_used,
            "action_log": action_log,
            "iterations": len(action_log),
            "query": user_query
        }
    
    def __del__(self):
        """Cleanup."""
        if hasattr(self, 'pg_client'):
            self.pg_client.close()
