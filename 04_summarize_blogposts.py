import math
from typing import Optional, Tuple
import os
from PIL import Image
from smolagents import CodeAgent, GoogleSearchTool, HfApiModel, VisitWebpageTool, LiteLLMModel, DuckDuckGoSearchTool
from smolagents import tool



model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=8192
)
# print(calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093)))

# task = """Find all Real World Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
# Also give me some supercar factories with the same cargo plane transfer time."""



agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    additional_authorized_imports=["pandas"],
    max_steps=20,
)

# Adding planing steps to the agent
agent.planning_interval = 4

if __name__ == "__main__":

    input_site = "https://towardsdatascience.com/3-challenges-of-data-adoption-790a87ae3472/"

    task = f"""
    Read the article on {input_site} using the tools at you disposal. Summarize the content of the blogpost extracting the key insights.
    """

    detailed_report = agent.run(f"""
    You're an expert analyst. You make comprehensive reports after visiting websites.
    You will get a url. Your Task is to write a summarization of the Key Points.
    For each summerisation you write, visit the source url to confirm numbers.

    {task}
    """)

    print(detailed_report)
