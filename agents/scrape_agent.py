from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, tool
from source.ai_blogs import ai_blog_sitemaps
from tools.scrape_website import scrape_website_using_sitemap_url

model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=8192
)

agent = CodeAgent(tools=[scrape_website_using_sitemap_url],
                  model=model,
                  additional_authorized_imports=["asyncio"]
                  )

test_page = ["https://towardsdatascience.com/post-sitemap.xml",
             "https://openai.com/sitemap.xml/research/"
             ]

agent.run(
    f"""
    scrape the following sites one after another using the tools you got at hand.
    {test_page}
    The tool will skip urls that where scraped before. So, if nothing is scraped you are up to date
    and everything went fine.
     
    You will get feedback from the tool, explain what the tool returned to the user.
    
    """
)


