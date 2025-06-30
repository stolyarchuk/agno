"""
🗓️ Pipedream Google Calendar MCP

This example shows how to use Pipedream MCP servers (in this case the Google Calendar one) with Agno Agents.

1. Connect your Pipedream and Google Calendar accounts: https://mcp.pipedream.com/app/google-calendar
2. Get your Pipedream MCP server url: https://mcp.pipedream.com/app/google-calendar
3. Set the MCP_SERVER_URL environment variable to the MCP server url you got above
4. Install dependencies: pip install agno mcp-sdk
"""

import asyncio
import os

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from agno.utils.log import log_exception

mcp_server_url = os.getenv("MCP_SERVER_URL")


async def run_agent(task: str) -> None:
    try:
        async with MCPTools(
            url=mcp_server_url, transport="sse", timeout_seconds=20
        ) as mcp:
            agent = Agent(
                model=OpenAIChat(id="gpt-4o-mini"),
                tools=[mcp],
                markdown=True,
            )
            await agent.aprint_response(message=task, stream=True)
    except Exception as e:
        log_exception(f"Unexpected error: {e}")


if __name__ == "__main__":
    asyncio.run(
        run_agent("Tell me about all events I have in my calendar for tomorrow")
    )
