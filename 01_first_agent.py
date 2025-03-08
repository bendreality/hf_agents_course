from random import randint

from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_model, tool, LiteLLMModel, OpenAIServerModel
import datetime
import requests
import pytz
import yaml

from tools.final_answer import FinalAnswerTool
from Gradio_UI import GradioUI

from tools.image_gen import queue_prompt


@tool
def roll_a_dice(die_sides: int) -> str:  # it's important to specify the return type
    # Keep this format for the tool description / args description but feel free to modify the tool
    """A tool that rolls a die.
    If not specified roll a 6 sided die
    Args:
        die_sides: how many sides the die has
    """

    roll_result = randint(1,die_sides)

    return f"you rolled a {roll_result}"


@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """A tool that fetches the current local time in a specified timezone.
    Args:
        timezone: A string representing a valid timezone (e.g., 'America/New_York').
    """
    try:
        # Create timezone object
        tz = pytz.timezone(timezone)
        # Get current time in that timezone
        local_time = datetime.datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")
        return f"The current local time in {timezone} is: {local_time}"
    except Exception as e:
        return f"Error fetching time for timezone '{timezone}': {str(e)}"


final_answer = FinalAnswerTool()
model = LiteLLMModel(
    model_id="ollama/qwen2.5:14b",  # This model is a bit weak for agentic behaviours though
    api_base="http://localhost:11434",  # replace with 127.0.0.1:11434 or remote open-ai compatible server if necessary
    api_key="YOUR_API_KEY",  # replace with API key if necessary
    num_ctx=8192,
    # ollama default is 2048 which will fail horribly. 8192 works for easy tasks, more is better. Check https://huggingface.co/spaces/NyxKrage/LLM-Model-VRAM-Calculator to calculate how much VRAM this will need for the selected model.
)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

# We're creating our CodeAgent
agent = CodeAgent(
    model=model,
    tools=[get_current_time_in_timezone, final_answer, roll_a_dice, queue_prompt],  # add your tools here (don't remove final_answer)
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)

GradioUI(agent).launch()
