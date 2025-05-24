import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.serpapi import SerpAPIWrapper
from data_manager import SatelliteDataManager
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables
load_dotenv()

# Get API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Define response schemas
response_schemas = [
    # Basic Information
    ResponseSchema(name="altitude", description="Satellite altitude in kilometers"),
    ResponseSchema(name="altitude_source", description="Source URL for altitude data"),
    ResponseSchema(name="orbital_life_years", description="Orbital life in years"),
    ResponseSchema(name="orbital_life_source", description="Source URL for orbital life data"),
    ResponseSchema(name="launch_orbit_classification", description="ISRO orbit classification (GTO, LEO, or SSO)"),
    ResponseSchema(name="orbit_classification_source", description="Source URL for orbit classification"),
    ResponseSchema(name="number_of_payloads", description="Number of payloads on the satellite"),
    ResponseSchema(name="payloads_source", description="Source URL for payload information")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

class BasicInfoBot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY
        )
        
        self.data_manager = SatelliteDataManager()

    def get_tools(self):
        search = TavilySearchResults(api_key=TAVILY_API_KEY)
        tavily_search = Tool(
            name="tavily_search",
            description="Search the web for basic satellite information",
            func=search.run,
            verbose=True
        )

        serpapi_search = Tool(
            name="serpapi_search",
            description="Search the web with SerpAPI for more comprehensive information",
            func=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run,
            verbose=True
        )

        return [tavily_search, serpapi_search]

    def get_prompt_template(self):
        template = """
        You are a satellite basic information researcher. Your task is to find specific basic information about the given satellite.

        Available tools:
        {tools}

        Tools names:
        {tool_names}

        IMPORTANT GUIDELINES:
        1. Focus ONLY on basic information:
           - Altitude
           - Orbital life
           - Orbit classification
           - Number of payloads
        2. Always verify information from multiple sources
        3. Include source URLs for each piece of information
        4. Be precise with numerical values

        CRITICAL INSTRUCTION:
        You must follow the ReAct format for your responses:
        Thought: (your reasoning about what to do next)
        Action: (the tool to use)
        Action Input: (the input for the tool)
        Observation: (the result of the action)
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        When you have all the information, end with:
        Thought: I now know the final answer
        Final Answer: {"altitude": "value or 'Not found'",
                      "altitude_source": "source URL or 'Not found'",
                      "orbital_life_years": "value or 'Not found'",
                      "orbital_life_source": "source URL or 'Not found'",
                      "launch_orbit_classification": "value or 'Not found'",
                      "orbit_classification_source": "source URL or 'Not found'",
                      "number_of_payloads": "value or 'Not found'",
                      "payloads_source": "source URL or 'Not found'"}

        DO NOT WAIT FOR COMPLETE INFORMATION. Return whatever data you have gathered, even if some fields are missing.
        Use "Not found" for any fields where you couldn't find information.

        {format_instructions}

        Question: {input}
        {agent_scratchpad}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad", "format_instructions"]
        )

    def process_satellite(self, satellite_name):
        """Process a satellite and store its basic information"""
        tools = self.get_tools()
        prompt = self.get_prompt_template()
        
        agent = create_react_agent(
            self.llm,
            tools,
            prompt
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=8,
            early_stopping_method="force"
        )

        try:
            # Create input dictionary with all expected variables
            input_dict = {
                "input": f"Find basic information about {satellite_name}",
                "tools": tools,
                "tool_names": [tool.name for tool in tools],
                "agent_scratchpad": "",
                "format_instructions": format_instructions
            }
            
            result = agent_executor.invoke(input_dict)
            try:
                parsed_output = output_parser.parse(result["output"])
            except Exception as parse_error:
                print(f"Error parsing output: {str(parse_error)}")
                print("Raw output:", result["output"])
                # Create a default structure with "Not found" values
                parsed_output = {
                    "altitude": "Not found",
                    "altitude_source": "Not found",
                    "orbital_life_years": "Not found",
                    "orbital_life_source": "Not found",
                    "launch_orbit_classification": "Not found",
                    "orbit_classification_source": "Not found",
                    "number_of_payloads": "Not found",
                    "payloads_source": "Not found"
                }
            
            self.data_manager.append_satellite_data(
                satellite_name,
                "basic_info",
                parsed_output
            )
            return parsed_output
        except Exception as e:
            print(f"Error processing satellite {satellite_name}: {str(e)}")
            return None