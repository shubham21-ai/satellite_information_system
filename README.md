# 🛰️ Satellite Information System

A comprehensive web application that gathers and displays detailed information about satellites using specialized AI agents. The system collects data from various sources and presents it in an organized, user-friendly interface.

## 🌟 Features

- **Multi-Agent Information Gathering**
  - Basic Information (altitude, orbital life, etc.)
  - Technical Specifications (type, applications, sensors)
  - Launch and Cost Information

- **Interactive Web Interface**
  - Real-time data updates
  - Organized tabular display
  - Raw JSON data access
  - Data export functionality

- **Comprehensive Data Collection**
  - Multiple authoritative sources
  - Cross-referenced information
  - Source URLs for verification
  - Detailed technical specifications

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for:
  - Google Gemini AI
  - Tavily Search
  - SerpAPI

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd satellite_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```env
GOOGLE_API_KEY=your_google_api_key
TAVILY=your_tavily_api_key
SERPAPI_API_KEY=your_serpapi_api_key
```

### Running the Application

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## 📊 Usage

1. Enter a satellite name in the sidebar
2. The system will gather information using specialized AI agents
3. View the collected information in organized tabs:
   - Basic Information
   - Technical Specifications
   - Launch & Cost Information
   - Raw JSON Data

4. Download the data in JSON format for further analysis

## 🏗️ System Architecture

The system consists of three main components:

1. **BasicInfoBot**
   - Collects fundamental satellite data
   - Focuses on altitude, orbital life, and payload information

2. **TechnicalSpecsBot**
   - Gathers detailed technical specifications
   - Includes sensor data and technological breakthroughs

3. **LaunchCostBot**
   - Collects launch and cost-related information
   - Tracks mission costs and launch vehicle details

## 🔧 Data Management

- All satellite data is stored in `satellite_data.json`
- Data is automatically updated when new information is gathered
- Previous searches are saved for quick access
- Data can be exported in JSON format

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini AI for providing the AI capabilities
- Tavily and SerpAPI for search functionality
- Streamlit for the web interface framework 