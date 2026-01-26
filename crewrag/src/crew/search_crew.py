import yaml
from dotenv import load_dotenv
import os
from crewrag.src.tools.pdf_tool import QdrantSearchTool

# Load environment variables
load_dotenv()

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool

class SearchCrew:
    """SearchCrew for document search and research tasks"""
    
    # Agent configurations
    agents_config = {
        'search_agent': {
            'role': "Information Retrieval Specialist",
            'goal': (
                "Find the most relevant and accurate information to answer: {query}. "
                "ALWAYS search the PDF knowledge base first using the 'Search PDF Knowledge Base' tool. "
                "Only use web search if the PDF search returns no results or insufficient information."
            ),
            'backstory': (
                "You are a systematic researcher who follows a strict information retrieval protocol. "
                "You always check internal knowledge bases before searching external sources. "
                "You understand when local documentation has the answer versus when broader web search is needed."
            )
        },
        'formalize_response_agent': {
            'role': "Response Synthesizer",
            'goal': (
                "Create a clear, well-structured answer to: {query} "
                "based on the retrieved information. Include source citations."
            ),
            'backstory': (
                "You are an expert at synthesizing information from multiple sources "
                "into coherent, accurate responses. You always cite your sources properly."
            )
        }
    }
    
    # Task configurations
    tasks_config = {
        'search_task': {
            'description': (
                "Search for information to answer: {query}\n\n"
                "MANDATORY SEARCH PROTOCOL:\n"
                "1. FIRST: Use 'Search PDF Knowledge Base' tool with the query\n"
                "2. Evaluate the PDF search results:\n"
                "   - If relevant results found (score > 0.5): Proceed with that information\n"
                "   - If no results or low scores (< 0.5): Use 'Search the internet' tool\n"
                "3. Document which tool(s) you used and why\n\n"
                "Return all findings with source information."
            ),
            'expected_output': (
                "A comprehensive collection of relevant information including:\n"
                "- Direct excerpts from PDF knowledge base (if found)\n"
                "- Web search results (if needed)\n"
                "- Similarity scores and source attribution\n"
                "- Clear indication of which sources were used"
            )
        },
        'formalize_task': {
            'description': (
                "Synthesize a final answer for: {query}\n\n"
                "Based on the search results from the previous task:\n"
                "1. Combine information from all sources\n"
                "2. Create a clear, concise response\n"
                "3. Include citations showing which sources were used\n"
                "4. If information wasn't found, clearly state that"
            ),
            'expected_output': (
                "A well-formatted final response that:\n"
                "- Directly answers the query\n"
                "- Cites specific sources (PDF knowledge base or web)\n"
                "- Is clear and easy to understand\n"
                "- Indicates confidence level based on source quality"
            )
        }
    }

    def __init__(self):
        # Use Ollama with string-based model specification
        llm_model = os.getenv("LLM_MODEL", "llama3.2:1b")
        print(f"Using Ollama model: {llm_model}")
        
        # Store model string with ollama/ prefix
        self.model_name = f"ollama/{llm_model}"
        
        # Initialize tools
        self.search_tool = SerperDevTool()
        self.qdrant_tool = QdrantSearchTool(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
    
    def search_agent(self) -> Agent:
        return Agent(
            role=self.agents_config['search_agent']['role'],
            goal=self.agents_config['search_agent']['goal'],
            backstory=self.agents_config['search_agent']['backstory'],
            tools=[self.qdrant_tool, self.search_tool],
            llm=self.model_name,
            verbose=True,
            allow_delegation=False
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
            planning_enabled=True,
        )


def main():
    """Test the SearchCrew"""
    # Initialize the crew
    search_crew = SearchCrew()
    
    # Define test query
    # test_query = "What is the meaning of DSPY?"
    test_query = "What is the THE DSPY COMPILER. Summarize the stages please?"
    # Kickoff the crew with the query
    result = search_crew.crew().kickoff(inputs={'query': test_query})
    
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    print(result)
    
    return result


if __name__ == "__main__":
    main()