from smolagents import DuckDuckGoSearchTool, VisitWebpageTool, ToolCallingAgent, HfApiModel, CodeAgent

# Create web agent and manager agent structure
web_agent = ToolCallingAgent(
    tools=[DuckDuckGoSearchTool(),VisitWebpageTool()],
    model=HfApiModel(),         # Add model
    max_steps=10,        # Adjust steps
    name="web_agent",           # Add name
    description="searching te web for information"      # Add description
)

manager_agent = CodeAgent(
    model=HfApiModel(),
    managed_agents=[web_agent],
    max_steps=10

)


###
# Was gibt es über security zu wissen?
# Was ist die E2B Sandbox?
# Welche security settings gibt es?

# Set up secure code execution environment
from smolagents import CodeAgent, HfApiModel, VisitWebpageTool, E2BExecutor

agent = CodeAgent(
    tools=[VisitWebpageTool()],
    model=HfApiModel(),
    additional_authorized_imports=["requests", "markdownify"],
    use_e2b_executor=True
)

## Lösung
from smolagents import CodeAgent, E2BSandbox

agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    sandbox=E2BSandbox(),
    additional_authorized_imports=["numpy"]
)


#-------------------




# Create a tool-calling agent
from smolagents import ToolCallingAgent

agent = ToolCallingAgent(
    model=HfApiModel(),
    max_steps=10,
    name="tool_agent",
    description="calls tools using json",
    tools=[]
    # Add configuration here
)

#---- Lösung
from smolagents import ToolCallingAgent

agent = ToolCallingAgent(
    tools=[custom_tool],
    model=model,
    max_steps=5,
    name="tool_agent",
    description="Executes specific tools based on input"
)




#------


# Configure model integration
from smolagents import HfApiModel

model = HfApiModel(model_id="Qwen/QwQ-32B")


#----- Lösung
from smolagents import HfApiModel, LiteLLMModel

# Hugging Face model
hf_model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

# Alternative model via LiteLLM
other_model = LiteLLMModel("anthropic/claude-3-sonnet")