import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, Tool, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities import SerpAPIWrapper
from data_manager import SatelliteDataManager
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import streamlit as st
import os

load_dotenv()

# Get API keys from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")




# Define response schemas
response_schemas = [
    # Satellite Type
    ResponseSchema(name="satellite_type", description="Type of satellite (Communication/ Earth Observation / Experimental / Navigation / Science & Exploration)"),
    ResponseSchema(name="satellite_type_source", description="Source URL for satellite type"),
    
    # Satellite Application
    ResponseSchema(name="satellite_application", description="Detailed description of satellite application"),
    ResponseSchema(name="application_source", description="Source URL for satellite application"),
    
    # Sensor Specifications
    ResponseSchema(name="sensor_specs", description="JSON object containing spectral_bands and spatial_resolution"),
    ResponseSchema(name="sensor_specs_source", description="Source URL for sensor specifications"),
    
    # Technological Breakthroughs
    ResponseSchema(name="technological_breakthroughs", description="Notable technological breakthroughs"),
    ResponseSchema(name="breakthrough_source", description="Source URL for breakthrough information")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

class TechnicalSpecsBot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=2048
        )

        self.data_manager = SatelliteDataManager()

    def get_tools(self):
        search = TavilySearchResults(api_key=TAVILY_API_KEY)
        tavily_search = Tool(
            name="tavily_search",
            description="Search the web for technical specifications",
            func=search.run,
            verbose=True
        )

        serpapi_search = Tool(
            name="serpapi_search",
            description="Search the web with SerpAPI for more comprehensive information",
            func=SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY).run,
            verbose=True
        )
        ddg_search = Tool(
            name="duckduckgo_search",
            description="Alternative search engine. Use when other searches don't return sufficient results.",
            func=DuckDuckGoSearchRun().run,
            verbose=True
        )

        return [tavily_search, serpapi_search, ddg_search]

    def get_prompt_template(self):
        template = """
        You are a satellite technical specifications researcher. Your task is to find detailed technical information about the given satellite.

        Available tools:
        {tools}

        Tools names:
        {tool_names}

        IMPORTANT GUIDELINES:
        1. Focus ONLY on technical specifications:
           - Satellite type
           - Application details
           - Sensor specifications
           - Technological breakthroughs
        2. Look for detailed technical documentation
        3. Include source URLs for each piece of information
        4. Pay special attention to sensor specifications and technological innovations

        CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS EXACT FORMAT:
        For EVERY response, you must use this exact format:

        Thought: (explain what you're going to do next)
        Action: (MUST be one of these exact tool names: {tool_names})
        Action Input: (the search query or input for the tool)
        Observation: (the result from the tool)

        IMPORTANT RULES:
        1. You MUST use one of the available tools listed above
        2. NEVER use "None" or any other tool name not listed
        3. After getting results, analyze them and decide if you need more information
        4. If you need more information, use another tool
        5. If you have enough information, proceed to the Final Answer
        6. NEVER output JSON directly without the proper Thought/Action format
        7. ALWAYS use the Final Answer format when providing the complete information
        8. NEVER make conclusions in the Thought section without using a tool
        9. ALWAYS use a tool to verify information before concluding
        10. If information is not found, use multiple tools to confirm before concluding

        When you have all the information, end with:
        Thought: I now have all the required information
        Final Answer: {"satellite_type": "value or 'Not found'",
                      "satellite_type_source": "source URL or 'Not found'",
                      "satellite_application": "value or 'Not found'",
                      "application_source": "source URL or 'Not found'",
                      "sensor_specs": {
                          "spectral_bands": "value or 'Not found'",
                          "spatial_resolution": "value or 'Not found'"
                      },
                      "sensor_specs_source": "source URL or 'Not found'",
                      "technological_breakthroughs": "value or 'Not found'",
                      "breakthrough_source": "source URL or 'Not found'"}

        DO NOT WAIT FOR COMPLETE INFORMATION. Return whatever data you have gathered, even if some fields are missing.
        Use "Not found" for any fields where you couldn't find information.

        {format_instructions}

        Question: {input}
        {agent_scratchpad}
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            partial_variables={"format_instructions": format_instructions}
        )

    def process_satellite(self, satellite_name):
        """Process a satellite and store its technical specifications"""
        tools = self.get_tools()
        agent = create_react_agent(
            self.llm,
            tools,
            self.get_prompt_template()
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
            # Run the agent
            result = agent_executor.invoke({"input": f"Find technical specifications for {satellite_name}"})
            
            # Parse the output using StructuredOutputParser
            try:
                parsed_output = output_parser.parse(result["output"])
            except Exception as parse_error:
                print(f"Error parsing output: {str(parse_error)}")
                print("Raw output:", result["output"])
                # Create a default structure with "Not found" values
                parsed_output = {
                    "satellite_type": "Not found",
                    "satellite_type_source": "Not found",
                    "satellite_application": "Not found",
                    "application_source": "Not found",
                    "sensor_specs": {
                        "spectral_bands": "Not found",
                        "spatial_resolution": "Not found"
                    },
                    "sensor_specs_source": "Not found",
                    "technological_breakthroughs": "Not found",
                    "breakthrough_source": "Not found"
                }
            
            # Store the data
            self.data_manager.append_satellite_data(
                satellite_name,
                "technical_specs",
                parsed_output
            )
            
            return parsed_output
            
        except Exception as e:
            print(f"Error processing satellite {satellite_name}: {str(e)}")
            return None 