import yaml
from dotenv import load_dotenv
import os
from crewrag.src.tools.pdf_tool import QdrantSearchTool
from crewrag.src.tools.web_search_tool import WebSearchTool  # ✅ Add this

# Load environment variables
load_dotenv()

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

class SearchCrew:
    """SearchCrew for document search and research tasks"""
    
    # Agent configurations
    agents_config = {
        'search_agent': {
            'role': "Information Retrieval Specialist",
            'goal': (
                "Find accurate information about: {query}\n"
                "Use PDF knowledge base for technical topics (DSPY, ML, AI frameworks). "
                "Use web search for general knowledge, current events, people, sports, news."
            ),
            'backstory': (
                "You are an intelligent researcher who knows which tool to use. "
                "You check PDF knowledge base first for technical topics, "
                "then use web search for everything else or when PDF returns 'NOT FOUND'."
            )
        },
        'formalize_response_agent': {
            'role': "Response Synthesizer",
            'goal': "Create a clear answer to: {query} based on search results",
            'backstory': (
                "You synthesize information from any source "
                "into clear, accurate responses with proper citations. Also mention which "
                "tool provided the information (PDF or web search)."
            )
        }
    }
    
    # Task configurations
    tasks_config = {
        'search_task': {
            'description': (
                "Find information about: {query}\n\n"
                "SEARCH STRATEGY:\n"
                "1. For technical topics (DSPY, ML, AI): Use 'Search PDF Knowledge Base' first\n"
                "2. If PDF search returns 'NOT FOUND' OR for general queries: Use 'Search the Internet'\n"
                "3. Return the results from whichever tool provided relevant information\n\n"
                "Examples:\n"
                "- 'What is DSPY?' → Try PDF first\n"
                "- 'Who is Alcaraz?' → Use web search\n"
                "- 'Current news' → Use web search\n"
            ),
            'expected_output': "Relevant information with sources (PDF or web)"
        },
        'formalize_task': {
            'description': (
                "Create a clear answer for: {query}\n\n"
                "Based on search results:\n"
                "1. Synthesize the information\n"
                "2. Include source citations\n"
                "3. Make it easy to understand\n"
                "4. Mention which tool provided the information (PDF or web search)\n"
            ),
            'expected_output': "A clear answer with source citations"
        }
    }

    def __init__(self):
        # Use OpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"Using OpenAI model: {model}")
        self.model_name = model
        
        # ✅ Initialize BOTH tools
        self.qdrant_tool = QdrantSearchTool(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self.web_search_tool = WebSearchTool()  # ✅ Add web search
    
    def search_agent(self) -> Agent:
        return Agent(
            role=self.agents_config['search_agent']['role'],
            goal=self.agents_config['search_agent']['goal'],
            backstory=self.agents_config['search_agent']['backstory'],
            tools=[self.qdrant_tool, self.web_search_tool],  # ✅ Both tools
            llm=self.model_name,
            verbose=True,
            allow_delegation=False,
            max_iter=5  # Limit iterations
        )
    
    def formalize_response_agent(self) -> Agent:
        return Agent(
            role=self.agents_config['formalize_response_agent']['role'],
            goal=self.agents_config['formalize_response_agent']['goal'],
            backstory=self.agents_config['formalize_response_agent']['backstory'],
            llm=self.model_name,
            verbose=True,
            allow_delegation=False,
        )
    
    def search_task(self) -> Task:
        return Task(
            description=self.tasks_config['search_task']['description'],
            expected_output=self.tasks_config['search_task']['expected_output'],
            agent=self.search_agent()
        )
    
    def formalize_task(self) -> Task:
        return Task(
            description=self.tasks_config['formalize_task']['description'],
            expected_output=self.tasks_config['formalize_task']['expected_output'],
            agent=self.formalize_response_agent(),
            context=[self.search_task()]
        )
    
    def crew(self) -> Crew:
        """Creates the SearchCrew crew"""
        return Crew(
            agents=[self.search_agent(), self.formalize_response_agent()],
            tasks=[self.search_task(), self.formalize_task()],
            process=Process.sequential,
            verbose=True,
        )


def main():
    """Test the SearchCrew"""
    search_crew = SearchCrew()
    
    # Test both types of queries
    queries = [
        "What is DSPY?",  # Should use PDF
        "Who is Carlos Alcaraz?",  # Should use web search
    ]
    
    for test_query in queries:
        print(f"\n{'#'*80}")
        print(f"TESTING: {test_query}")
        print(f"{'#'*80}\n")
        
        result = search_crew.crew().kickoff(inputs={'query': test_query})
        
        print("\n" + "="*80)
        print("FINAL RESULT:")
        print("="*80)
        print(result)
        print("\n")
    
    return result


if __name__ == "__main__":
    main()