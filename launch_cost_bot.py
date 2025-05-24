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
    # Launch Cost Information
    ResponseSchema(name="launch_cost", description="Launch cost in USD"),
    ResponseSchema(name="launch_cost_source", description="Source URL for launch cost data"),
    ResponseSchema(name="launch_vehicle", description="Launch vehicle used"),
    ResponseSchema(name="launch_vehicle_source", description="Source URL for launch vehicle information"),
    ResponseSchema(name="launch_date", description="Launch date"),
    ResponseSchema(name="launch_date_source", description="Source URL for launch date information"),
    ResponseSchema(name="launch_site", description="Launch site"),
    ResponseSchema(name="launch_site_source", description="Source URL for launch site information"),
    
    # Launch Mass
    ResponseSchema(name="launch_mass", description="JSON object containing max_leo and actual_mass"),
    ResponseSchema(name="launch_mass_source", description="Source URL for launch mass information"),
    
    # Launch Success
    ResponseSchema(name="launch_success", description="Launch success status (1 for success, 0 for failure)"),
    ResponseSchema(name="launch_success_source", description="Source URL for launch success information"),
    
    # Vehicle Reusability
    ResponseSchema(name="vehicle_reusability", description="Vehicle reusability status (1 for reusable, 0 for not)"),
    ResponseSchema(name="reusability_details", description="Details about vehicle reusability"),
    ResponseSchema(name="reusability_source", description="Source URL for reusability information"),
    
    # Mission Cost
    ResponseSchema(name="mission_cost", description="JSON object containing all cost components"),
    ResponseSchema(name="mission_cost_source", description="Source URL for mission cost information")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

class LaunchCostBot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=4098
        )

        self.data_manager = SatelliteDataManager()

    def get_tools(self):
        search = TavilySearchResults(api_key=TAVILY_API_KEY)
        
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
        tavily_search = Tool(
            name="tavily_search",
            description="Search the web for launch and cost information",
            func=search.run,
            verbose=True
        )

        def space_industry_search(query):
            """Search specifically in space industry sources"""
            specialized_sites = [
                "site:spacenews.com",
                "site:spaceflightnow.com", 
                "site:spacex.com",
                "site:nasa.gov",
                "site:esa.int",
                "site:spacepolicyonline.com",
                "site:satellitetoday.com"
            ]
            enhanced_query = f"{query} ({' OR '.join(specialized_sites)})"
            return tavily_search.run(enhanced_query)

        space_search_tool = Tool(
            name="space_industry_search",
            description="Search specialized space industry websites for authoritative satellite information.",
            func=space_industry_search,
            verbose=True
        )

        def financial_search(query):
            """Search financial and investment sources"""
            financial_sites = [
                "site:reuters.com",
                "site:bloomberg.com",
                "site:cnbc.com",
                "site:sec.gov",
                "site:investor.com"
            ]
            enhanced_query = f"{query} cost budget funding ({' OR '.join(financial_sites)})"
            return tavily_search.run(enhanced_query)

        financial_search_tool = Tool(
            name="financial_search",
            description="Search financial news and SEC filings for satellite mission costs and budgets.",
            func=financial_search,
            verbose=True
        )

        def technical_specs_search(query):
            """Search for technical specifications"""
            tech_sites = [
                "site:wikipedia.org",
                "site:gunterspace.com",
                "site:skyrocket.de",
                "site:rocketrundown.com"
            ]
            enhanced_query = f"{query} specifications mass launch vehicle ({' OR '.join(tech_sites)})"
            return tavily_search.run(enhanced_query)

        tech_search_tool = Tool(
            name="technical_specs_search",
            description="Search technical databases and specifications for satellite mass and launch vehicle details.",
            func=technical_specs_search,
            verbose=True
        )

        return [tavily_search, serpapi_search, ddg_search, space_search_tool, financial_search_tool, tech_search_tool]

    def get_prompt_template(self):
        template = """
        You are a satellite launch and cost information researcher. Your task is to find detailed information about the launch and costs of the given satellite.

        Available tools:
        {tools}

        Tools names:
        {tool_names}

        IMPORTANT GUIDELINES:
        1. Focus ONLY on launch and cost information:
           - Launch mass details
           - Launch success/failure
           - Vehicle reusability
           - All cost components
        2. Look for official documentation and news sources
        3. Include source URLs for each piece of information
        4. Pay special attention to cost breakdowns and launch specifications

        SEARCH STRATEGY - FOLLOW THIS EXACTLY:
        1. Search for launch mass details
        2. Search financial/cost information from multiple angles
        3. Search technical specifications and mass data
        4. Verify information across multiple sources
        5. If data is missing, try alternative search terms and sources

        CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS EXACT FORMAT:
        For EVERY response, you must use this exact format:

        Thought: (explain what you're going to do next)
        Action: (choose one of the available tools: {tool_names})
        Action Input: (the search query or input for the tool)
        Observation: (the result from the tool)

        Repeat this Thought/Action/Action Input/Observation pattern as needed.

        When you have all the information, end with:
        Thought: I now have all the required information
        Final Answer: {{
            "launch_cost": "value or 'Not found'",
            "launch_cost_source": "source URL or 'Not found'",
            "launch_vehicle": "value or 'Not found'",
            "launch_vehicle_source": "source URL or 'Not found'",
            "launch_date": "value or 'Not found'",
            "launch_date_source": "source URL or 'Not found'",
            "launch_site": "value or 'Not found'",
            "launch_site_source": "source URL or 'Not found'",
            "launch_mass": {{
                "max_leo": "value or 'Not found'",
                "actual_mass": "value or 'Not found'"
            }},
            "launch_mass_source": "source URL or 'Not found'",
            "launch_success": "value or 'Not found'",
            "launch_success_source": "source URL or 'Not found'",
            "vehicle_reusability": "value or 'Not found'",
            "reusability_details": "value or 'Not found'",
            "reusability_source": "source URL or 'Not found'",
            "mission_cost": {{
                "overall_cost": "value or 'Not found'",
                "vehicle_cost": "value or 'Not found'",
                "development_cost": "value or 'Not found'",
                "approved_cost": "value or 'Not found'",
                "operational_cost": "value or 'Not found'"
            }},
            "mission_cost_source": "source URL or 'Not found'"
        }}

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
        """Process a satellite and store its launch and cost information"""
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
            result = agent_executor.invoke({"input": f"Find launch and cost information for {satellite_name}"})
            
            # Parse the output using StructuredOutputParser
            try:
                parsed_output = output_parser.parse(result["output"])
            except Exception as parse_error:
                print(f"Error parsing output: {str(parse_error)}")
                print("Raw output:", result["output"])
                # Create a default structure with "Not found" values
                parsed_output = {
                    "launch_cost": "Not found",
                    "launch_cost_source": "Not found",
                    "launch_vehicle": "Not found",
                    "launch_vehicle_source": "Not found",
                    "launch_date": "Not found",
                    "launch_date_source": "Not found",
                    "launch_site": "Not found",
                    "launch_site_source": "Not found",
                    "launch_mass": {
                        "max_leo": "Not found",
                        "actual_mass": "Not found"
                    },
                    "launch_mass_source": "Not found",
                    "launch_success": "Not found",
                    "launch_success_source": "Not found",
                    "vehicle_reusability": "Not found",
                    "reusability_details": "Not found",
                    "reusability_source": "Not found",
                    "mission_cost": {
                        "overall_cost": "Not found",
                        "vehicle_cost": "Not found",
                        "development_cost": "Not found",
                        "approved_cost": "Not found",
                        "operational_cost": "Not found"
                    },
                    "mission_cost_source": "Not found"
                }
            
            # Store the data
            self.data_manager.append_satellite_data(
                satellite_name,
                "launch_cost_info",
                parsed_output
            )
            
            return parsed_output
            
        except Exception as e:
            print(f"Error processing satellite {satellite_name}: {str(e)}")
            return None 