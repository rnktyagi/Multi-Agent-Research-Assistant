import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search_tool(query: str) -> str:
    """
    Searches the web for the given query to retrieve the latest, 
    highly relevant, and concise context for research purposes.
    """
    # 1. Dynamically grab the current year and date
    current_year = datetime.now().year
    current_date = datetime.now().strftime("%B %d, %Y")

    # 2. Query Augmentation: Force Tavily to look for recent data
    # If the LLM didn't specify a year, we secretly append it to the search
    if str(current_year) not in query:
        enhanced_query = f"{query} {current_year} latest"
    else:
        enhanced_query = query

    # 3. Execute the search
    response = tavily_client.search(
        query=enhanced_query, 
        max_results=5, 
        search_depth="advanced"
    )
    
    results = response.get("results", [])
        
    # 4. Context Anchoring: Inject the exact current date at the very top of the payload
    # This prevents the LLM from drifting back to its 2023 training weights
    final_response = [
        f"[SYSTEM NOTE: Today's exact date is {current_date}. Base all your synthesis, timelines, and citations relative to this present date.]\n"
    ]
    
    for result in results:
        title = result.get("title", "Untitled")
        source = result.get("url", "No URL available")
        content = result.get("content", "")
        
        # Tavily often returns a 'published_date'. If it doesn't, we mark it as Unknown.
        published_date = result.get("published_date", "Unknown Date")
        
        final_response.append(
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Published: {published_date}\n"
            f"Snippet: {content}\n"
            f"---"
        )

    # Compile the final string
    output_string = "\n\n".join(final_response)
    
    # 5. Fixed the print statement so it cleanly prints to your backend terminal
    print(f"\n🔍 SEARCH EXECUTED: '{enhanced_query}'\n{output_string}\n")
    
    return output_string