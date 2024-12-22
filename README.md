# CHAT SUMMARIZER

CHAT 🗣️ SUMMARY 📈 ANALYSIS

**Chat Summarizer** is a comprehensive web application designed to analyze and summarize chat data from messaging platforms. The tool provides insights into user communication patterns, keywords, and topics of conversation, along with graphical visualizations and a downloadable PDF report.

## Features

- **Chat Parsing**: Extracts structured data (date, time, sender, and messages) from raw chat text files.
- **Preprocessing**: Cleans and preprocesses messages to remove stopwords, punctuation, and irrelevant information.
- **Filtering**: Apply date, time, and keyword-based filters to refine the chat data for analysis.
- **Keyword Extraction**: Identifies and ranks the most significant keywords in the chat.
- **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to identify key topics of conversation.
- **Visualizations**: Interactive charts for message trends, topic distributions, and keyword importance.
- **PDF Report Generation**: Exports the analysis results, including charts and message summaries, into a structured PDF report.

## Project Structure

```
CHAT SUMMARIZER/

├── app.py                  🚀 Main application file

├── summarizer.ipynb        📝 Jupyter notebook for summarization

├── requirements.txt        📋 Required dependencies

├── Data/                   📊 Data Directory

    └── Chats.txt           💬 Raw dummy data chat file

└── Report/                 📑 Analysis Reports and Visualizations

    ├── chat_analysis_report.pdf   📄 PDF report from Jupyter notebook

    ├── keywords_plot.png          📊 Keywords analysis plot

    ├── message_trends_plot.png    📈 Message trends visualization

    ├── report.pdf                 📄 PDF report from web application

    ├── top_keywords.png           📊 Top keywords bar plot

    └── topics_plot.png            🧩 Topics distribution pie chart
```

## Getting Started

### Prerequisites
- Python 3.8 or later.
- pip (Python package manager).
- Windows 10/11.
- Web Browser.

### Installation

#### With Windows Executable

1. Download the setup executable from release section of this repository.

2. Install the app via the setup.

3. Run the executable of the app, after the app is installed.

4. Open the application in your browser at [http://localhost:8501](http://localhost:8501).

5. If the PC running the web app is connected to a router, you can use your mobile device to launch the app by opening your mobile browser and launching the "Network URL :" given in the terminal. This will automatically use your PC as a server and your phone as a client. You can do this with as many other PCs and mobile devices as the PC running the app can support.

#### With Python

1. Clone this repository:
   ```bash
   git clone https://github.com/N-Elmer/CHAT-SUMMARIZER.git
   cd CHAT-SUMMARIZER
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run app.py
   ```

4. Open the application in your browser at [http://localhost:8501](http://localhost:8501).

5. If the PC running the web app is connected to a router, you can use your mobile device to launch the app by opening your mobile browser and launching the "Network URL :" given in the terminal. This will automatically use your PC as a server and your phone as a client. You can do this with as many other PCs and mobile devices as the PC running the app can support.

## How to Use

### WhatsApp Chat Export

#### Export Chat to TXT File
1. Open your WhatsApp.
2. DON'T SELECT MEDIA for exporting.
3. Export your chats to a text file.

### Telegram Chat Export

#### Export Chat to JSON File
1. Open your Telegram.
2. DON'T SELECT MEDIA for exporting.
3. Export your chats to a json file.

### Upload Chat File
1. With the sidebar, upload your exported chat file.
2. Use the filters on the sidebar to get more insights.

#### Apply Filters
- **Date and Time Filters**: Set start and end dates/times.
- **Keywords**: Enter comma-separated keywords to filter messages containing specific terms.

#### Generate Summary
- Select the number of top keywords, topics, and messages to include in the analysis.
- View visualizations, including:
  - **Message Trends**: A time-based line chart of message counts.
  - **Keyword Importance**: A bar chart of extracted keywords.
  - **Topic Distribution**: A pie chart of conversation topics.

#### Download Report
- Click "Generate Report" to create a PDF file summarizing the analysis, including:
  - Top keywords chart.
  - Message trends chart.
  - Summary of top messages.

## Key Features Explained

### Chat Parsing
The app processes chat text files to extract structured data into a DataFrame. It identifies timestamps, senders, and message contents while handling multiline messages and system notifications.

### Preprocessing
Cleans chat messages by:
- Removing stopwords (e.g., "the", "and").
- Tokenizing text.
- Eliminating punctuation.

### Topic Modeling
Utilizes Latent Dirichlet Allocation (LDA) to group conversations into topics. Each topic is represented by a set of key terms.

### Visualizations
Interactive charts provide actionable insights, making it easier to understand chat trends and topics.

## Dependencies

The following Python libraries are required:
- **streamlit**: Web app framework.
- **pandas**: Data manipulation.
- **plotly**: Interactive visualizations.
- **nltk**: Natural Language Toolkit for text preprocessing.
- **gensim**: Topic modeling.
- **reportlab**: PDF report generation.
- **seaborn**: Statistical data visualization.

Install all dependencies using:
```bash
pip install -r requirements.txt
```

## Troubleshooting

- **File Upload Issues**: Ensure the chat file is in `.txt` format and properly structured.
- **Date/Time Errors**: Verify that the date and time formats in the file match `DD/MM/YYYY` and `HH:MM AM/PM`.
- **PDF Generation Problems**: Ensure that the `Report/` directory exists and is writable.
- **Missing Dependencies**: Reinstall required packages using `pip install -r requirements.txt`.

## Contributing

Contributions are welcome! If you find a bug or have a feature request, please open an issue in the GitHub repository.

---

This README file provides an overview of the CHAT-SUMMARIZER web application, its folder structure, usage instructions, code explanation, and troubleshooting tips. Use it as a guide to understand and utilize the CHAT-SUMMARIZER app.

---
