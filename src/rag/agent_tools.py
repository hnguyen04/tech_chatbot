"""
Tool definitions for AgenticRAG pipeline.
"""
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional


@dataclass
class Tool:
    """Definition of a tool that the agent can use."""
    name: str
    description: str
    parameters: Dict[str, str]


# Tool definitions (functions are injected at runtime with dependencies)
TOOL_DEFINITIONS: List[Tool] = [
    Tool(
        name="search_products",
        description="Semantic search for products by description. Use for general product discovery based on features, use cases, or requirements.",
        parameters={"query": "search text describing what you're looking for", "top_k": "number of results (default 10)"}
    ),
    Tool(
        name="get_product_specs",
        description="Get detailed specifications for a specific product by its ID. Use when you need full specs for a product you already found.",
        parameters={"product_id": "integer product ID from a previous search"}
    ),
    Tool(
        name="compare_products",
        description="Get specs of multiple products for side-by-side comparison. Use when user wants to compare specific products.",
        parameters={"product_ids": "list of product IDs to compare (max 5)"}
    ),
    Tool(
        name="keyword_search",
        description="Exact keyword/model name search for PRODUCTS. Use for specific product names like 'iPhone 15 Pro Max' or exact model numbers. Returns actual product specs, not news articles.",
        parameters={"query": "exact product name or model number"}
    ),
]


def format_tools_for_prompt(tools: List[Tool] = TOOL_DEFINITIONS) -> str:
    """
    Format tool definitions for inclusion in LLM prompt.
    
    Args:
        tools: List of Tool definitions
        
    Returns:
        Formatted string describing available tools
    """
    lines = []
    for tool in tools:
        params_str = ", ".join([f"{k}: {v}" for k, v in tool.parameters.items()])
        lines.append(f"- {tool.name}({params_str})")
        lines.append(f"  Description: {tool.description}")
    return "\n".join(lines)


class ToolExecutor:
    """
    Executes tools with injected dependencies.
    """
    
    def __init__(self, retriever, pg_client):
        """
        Initialize tool executor with dependencies.
        
        Args:
            retriever: VietnameseRetriever instance for vector search
            pg_client: PostgresClient instance for database queries
        """
        self.retriever = retriever
        self.pg_client = pg_client
        
        # Build tool function map
        self._tools: Dict[str, Callable] = {
            "search_products": self._search_products,
            "get_product_specs": self._get_product_specs,
            "compare_products": self._compare_products,
            "keyword_search": self._keyword_search,
        }
    
    def execute(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            params: Parameters for the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self._tools:
            return {"error": f"Unknown tool: {tool_name}"}
        
        try:
            result = self._tools[tool_name](**params)
            return {"success": True, "data": result}
        except Exception as e:
            return {"error": str(e)}
    
    def _search_products(self, query: str, top_k: int = 10) -> List[Dict]:
        """Semantic search for products with full metadata."""
        docs = self.retriever.retrieve(query, k=top_k)
        results = []
        for doc in docs:
            # Include full metadata for source building in _generate_answer
            results.append({
                "record_id": doc.metadata.get("record_id"),
                "id": doc.metadata.get("record_id"),  # Alias for compatibility
                "content": doc.page_content[:500],  # Truncate for context
                "content_text": doc.page_content,  # Full content for answer generation
                "full_title": doc.metadata.get("title", ""),
                "brand": doc.metadata.get("brand", ""),
                "model": doc.metadata.get("model", ""),
                "price": doc.metadata.get("price", ""),
                "source_url": doc.metadata.get("source_url", ""),
                "category": doc.metadata.get("category", ""),
                "score": doc.metadata.get("score", 0)
            })
        return results
    
    def _get_product_specs(self, product_id: int) -> Optional[Dict]:
        """Get product with full specifications - formatted for LLM."""
        product = self.pg_client.fetch_product_with_specs(product_id)
        if product and product.get('specifications'):
            specs_list = product['specifications']
            if isinstance(specs_list, list):
                # Format specs as readable text
                spec_text = "\n".join([f"- {s['standardized_key']}: {s['standardized_value']}" for s in specs_list])
                product['specifications'] = spec_text
                product['specs_list'] = specs_list
        return product
    
    def _compare_products(self, product_ids: List[int]) -> List[Dict]:
        """Compare multiple products - returns formatted specs for LLM."""
        results = self.pg_client.compare_products(product_ids)
        
        # Format specifications as readable text for each product
        for product in results:
            if product and product.get('specifications'):
                specs_list = product['specifications']
                if isinstance(specs_list, list):
                    spec_text = "\n".join([f"- {s['standardized_key']}: {s['standardized_value']}" for s in specs_list])
                    product['specifications'] = spec_text
                    product['specs_list'] = specs_list
        
        return results
    
    def _keyword_search(self, query: str) -> List[Dict]:
        """Keyword search for exact matches - prioritizes actual products over articles."""
        print(f"    [keyword_search] Query: '{query}'")
        
        # Use products_only=True to get actual product specs, not news articles
        results = self.pg_client.search_keyword(query, limit=10, products_only=True)
        print(f"    [keyword_search] Products found: {len(results) if results else 0}")
        
        # If no products found, fall back to all results (articles included)
        if not results:
            print(f"    [keyword_search] Falling back to all content types...")
            results = self.pg_client.search_keyword(query, limit=10, products_only=False)
            print(f"    [keyword_search] All content found: {len(results) if results else 0}")
        
        # Convert to list of dicts and include full specs
        output = []
        for r in results:
            item = dict(r)
            # Fetch specs for products
            if item.get('content_type') == 'product' and item.get('id'):
                specs = self.pg_client.fetch_product_specs(item['id'])
                if specs:
                    # Format specs as readable text
                    spec_text = "\n".join([f"- {s['standardized_key']}: {s['standardized_value']}" for s in specs])
                    item['specifications'] = spec_text
                    item['specs_list'] = specs
            output.append(item)
        
        if output:
            print(f"    [keyword_search] Returning {len(output)} items, first: {output[0].get('full_title', 'N/A')[:50]}")
        else:
            print(f"    [keyword_search] No results found for '{query}'")
        
        return output if output else []
