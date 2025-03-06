from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.agentql import AgentQLTools

# Create agent with default AgentQL tool
agent = Agent(
    model=OpenAIChat(id="gpt-4o"), tools=[AgentQLTools()], show_tool_calls=True
)
agent.print_response("https://docs.agno.com/introduction", markdown=True)

# Define custom AgentQL query for specific data extraction
custom_query = """
{
    title
    text_content[]
}
"""

# Create AgentQL tool with custom query
custom_scraper = AgentQLTools(agentql_query=custom_query, custom_scrape=True)

# Create agent with custom AgentQL tool
custom_agent = Agent(
    model=OpenAIChat(id="gpt-4o"), tools=[custom_scraper], show_tool_calls=True
)
custom_agent.print_response("https://docs.agno.com/introduction", markdown=True)
