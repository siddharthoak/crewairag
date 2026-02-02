from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from typing import Any

class WebSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo"""
    name: str = "Search the Internet"
    description: str = (
        "Searches the internet for current information using DuckDuckGo. "
        "Use this when PDF knowledge base returns 'NOT FOUND' or for queries about: "
        "current events, people, sports, news, general knowledge not in PDFs. "
        "Input: search query as a string."
    )
    
    def _run(self, query: str) -> str:
        """
        Search the web
        
        Args:
            query: Search query string
            
        Returns:
            Formatted search results
        """
        try:
            print("\n" + "="*80)
            print("🌐 WEB SEARCH TOOL CALLED")
            print("="*80)
            print(f"📥 QUERY: {query}")
            print("="*80 + "\n")
            
            # Search with DuckDuckGo
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
            
            if not results:
                return "No web results found for this query."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body = result.get('body', 'No description')
                url = result.get('href', 'No URL')
                
                print(f"🔗 Result {i}:")
                print(f"   📌 {title}")
                print(f"   📝 {body[:150]}...")
                print(f"   🌐 {url}\n")
                
                formatted_results.append(
                    f"Result {i}:\n"
                    f"Title: {title}\n"
                    f"Description: {body}\n"
                    f"URL: {url}\n"
                )
            
            output = "\n".join(formatted_results)
            
            print("="*80)
            print("📤 WEB SEARCH OUTPUT:")
            print("="*80)
            print(output[:600])
            if len(output) > 600:
                print("...(truncated)")
            print("="*80 + "\n")
            
            return output
            
        except Exception as e:
            error_msg = f"Error in web search: {str(e)}"
            print(f"\n❌ {error_msg}\n")
            return error_msg