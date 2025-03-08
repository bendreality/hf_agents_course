import math
from dataclasses import dataclass
from typing import Optional, Tuple
import os
from PIL import Image
from smolagents import CodeAgent, GoogleSearchTool, HfApiModel, VisitWebpageTool, LiteLLMModel, DuckDuckGoSearchTool
from smolagents import tool

model_qwen_coder = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=16384
)

model_r1 = LiteLLMModel(
    model_id="ollama/deepseek-r1:14b",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=16384
)

model_llama32_vision = LiteLLMModel(
    model_id="ollama/llama3.2-vision",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=16384
)

model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=8192
)
# print(calculate_cargo_travel_time((41.8781, -87.6298), (-33.8688, 151.2093)))

# task = """Find all Real World Batman filming locations in the world, calculate the time to transfer via cargo plane to here (we're in Gotham, 40.7128° N, 74.0060° W), and return them to me as a pandas dataframe.
# Also give me some supercar factories with the same cargo plane transfer time."""

@dataclass
class DataPoint:
    contact_name: str = None
    beruf: str = None
    adresse: str = None
    mail_adresse: str = None



#-------- tooooools

class CustomizableGoogleSearchTool(GoogleSearchTool):
    name = "customizable_web_search"
    description = """Performs a google web search for your query then returns a string of the top search results.
    Allows customizing the number of results to display."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of search results to return",
            "default": 4,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi", default_max_results: int = 4):
        super().__init__(provider=provider)
        self.default_max_results = default_max_results

    def forward(self, query: str, filter_year: Optional[int] = None, max_results: Optional[int] = None) -> str:
        # Verwende den Standardwert, wenn max_results nicht angegeben ist
        if max_results is None:
            max_results = self.default_max_results

        # Stelle sicher, dass max_results eine positive ganze Zahl ist
        if max_results <= 0:
            max_results = self.default_max_results

        # Rufe die übergeordnete Methode auf, um die Suchergebnisse zu erhalten
        all_results = super().forward(query, filter_year)

        # Wenn keine Ergebnisse zurückgegeben wurden, gib die Nachricht unverändert zurück
        if all_results.startswith("No results found"):
            return all_results

        # Trenne den Header von den tatsächlichen Ergebnissen
        header = "## Search Results\n"
        results_list = all_results.replace(header, "").split("\n\n")

        # Begrenze die Anzahl der Ergebnisse
        limited_results = results_list[:max_results]

        # Setze die Ausgabe wieder zusammen
        return header + "\n\n".join(limited_results)

@tool
def write_to_markdown(text: str, filename:str) -> None:
    """
    takes in a text and writes it into a markdown file

    Args:
        text: The text that will be written into the markdown file
        filename: the name of the file

    Returns:
        bool: True, wenn erfolgreich geschrieben wurde, sonst False
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Text wurde erfolgreich in '{filename}' geschrieben.")
        return True
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei: {e}")
        return False


# Beispiel für die Verwendung:
# text = "# Überschrift\n\nDies ist ein **Beispieltext** in Markdown."
# write_to_markdown(text, "beispiel.md")

data_list = [
    # [1,"Herr Groth", "Selbständiger Elektromeister", "Peter-Joseph-Lenne-Str. 4, 51377 Leverkusen, Manfort", "5groth@web.de"],
    # [2,"Herr Agushi", "Übernahme Pizzeria SWerk Leverkusen" , "Ehrlichstr. 53, 51373 Leverkusen, Wiesdorf",	"besiboy_de@hotmail.de"],
    # [3,"Herr Karp", "","Bergische Landstr. 67, 51375 Leverkusen Schlebusch", ""],
    [4,"Herr Lupo",""," Hitdorfer Str. 205, 51371 Leverkusen", ""],
    [5,"","","Bergische Landstr. 67, 51375 Leverkusen, Schlebusch",""],
    [6,"","Creditreform","Hitdorfer Str. 205, 51371 Leverkusen",""],
    [7,"","Creditreform 02/2002", "Bergische Landstr. 78, 51375 Leverkusen",""],
    [8,"","Kfz Aufbereitung","Höhenstr. 38, 51381 Leverkusen", "kuehnmi5@gmail.com"],
    [9,"ulrich dost","Bayer 04 Leverkusen Fußball GmbH","122-124, 51373 Leverkusen"]
]


agent = CodeAgent(
    model=model_qwen_coder,
    tools=[CustomizableGoogleSearchTool(provider="serper",default_max_results=6), VisitWebpageTool(), write_to_markdown],
    additional_authorized_imports=[],
    max_steps=20,
)




# Adding planing steps to the agent
agent.planning_interval = 4

if __name__ == "__main__":

    input_site = "https://towardsdatascience.com/3-challenges-of-data-adoption-790a87ae3472/"

    for item in data_list:

        task = f"""
        ID: {item[0]}
        person associated with company: {item[1]}
        company: {item[2]}
        address: {item[3]}
        email: {item[4]}
        """

        detailed_report = agent.run(f"""
        You're an expert analyst. You check companys and persons by visiting websites.
        You will get some incomplete datasets. Your Task is to check if a company exists or not.
        The data an be bad and you have to be creative in your search. 
        Reason about what you got and what you can do with it. 
        
        Sometimes there will be no Company name, only an address. If so check if a company is located there.
        Search the web and check if the data you get is correct using the tools at you disposal.
        
        the companies are all in leverkusen germany
        always build your search around the address, combine that with other informations you got.
    
        
        The last step is to write the information you gathered into a markdown file. Use The ID as name for the file.
        
    
        {task}
        
        
        Output Format:
        
        ## Analysis of Company Existence at 
        
        ### Provided Information
        - **ID**: 
        - **Name of contact**: 
        - **Address**: 
        - **Email**: 
        
        ### Analysis
        - **Verification on Google Maps**:
        - **Search for Companies Located at the Specific Address**: 
        - **Business Registration Information**: 
        
        ### Conclusion
        
        
        ## Final Answer


        """)


