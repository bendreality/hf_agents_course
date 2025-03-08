from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, tool
from source.ai_blogs import ai_blog_sitemaps

model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=8192
)



agent = CodeAgent(tools=[],
                  model=model,
                  additional_authorized_imports=["selenium",
                                                 "selenium.webdriver.common.by",
                                                 "selenium.webdriver.common.keys",
                                                 "requests",
                                                 "subprocess"]
                  )

agent.run(
    """
    write code to use selenium to look if there is a sitemap on the webpage: https://bair.berkeley.edu
    try different approaches and names for the sitemap. Think about what is best practice first. 
    try requests first. If it's not working, use selenium.
    your task is to return the complete web adress to the sitemap.
    """
)
