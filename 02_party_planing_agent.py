from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, tool



model = LiteLLMModel(
    model_id = "ollama/qwen2.5-coder:14b",
    api_base = "http://localhost:11434",
    api_key = "noone needs an api key",
    num_ctx = 8192
)


# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party. available are casual, formal and superhero
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


agent = CodeAgent(tools=[], model=model, additional_authorized_imports=['datetime'])

agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes
    
    add some additional time to switch tasks

    If we start right now, at what time will the party be ready?
    """
)
