CHAT 🗣️ SUMMARY 📈 ANALYSIS
=============================
=============================

ABOUT
=======
WEBSITE: https://github.com/N-Elmer/CHAT-SUMMARIZER/

Chat Summarizer is a comprehensive web application designed to analyze and summarize chat data from messaging platforms. The tool provides insights into user communication patterns, keywords, and topics of conversation, along with graphical visualizations and a downloadable PDF report.

Features
=========
- Chat Parsing: Extracts structured data (date, time, sender, and messages) from raw chat text files.
- Preprocessing: Cleans and preprocesses messages to remove stopwords, punctuation, and irrelevant information.
- Filtering: Apply date, time, and keyword-based filters to refine the chat data for analysis.
- Keyword Extraction: Identifies and ranks the most significant keywords in the chat.
- Topic Modeling: Uses Latent Dirichlet Allocation (LDA) to identify key topics of conversation.
- Visualizations: Interactive charts for message trends, topic distributions, and keyword importance.
- PDF Report Generation: Exports the analysis results, including charts and message summaries, into a structured PDF report.

Getting Started
================

Prerequisites
--------------
- Windows 10/11.
- Web Browser.

Installation
=============

With Windows Executable
------------------------
1. Download the setup executable from release section of this repository.
2. Install the app via the setup.
3. Run the executable of the app, after the app is installed.
4. Open the application in your browser at http://localhost:8501/.
5. If the PC running the web app is connected to a router, you can use your mobile device to launch the app by opening your mobile browser and launching the "Network URL :" given in the terminal. This will automatically use your PC as a server and your phone as a client. You can do this with as many other PCs and mobile devices as the PC running the app can support.

How to Use
===========

WhatsApp Chat Export
---------------------

Export Chat to TXT File
------------------------
1. Open your WhatsApp.
2. DON'T SELECT MEDIA for exporting.
3. Export your chats to a text file.

Telegram Chat Export
---------------------

Export Chat to JSON File
-------------------------
1. Open your Telegram.
2. DON'T SELECT MEDIA for exporting.
3. Export your chats to a json file.

Upload Chat File
=================
1. With the sidebar, upload your exported chat file.
2. Use the filters on the sidebar to get more insights.

Apply Filters
==============
- Date and Time Filters: Set start and end dates/times.
- Keywords: Enter comma-separated keywords to filter messages containing specific terms.

Generate Summary
=================
- Select the number of top keywords, topics, and messages to include in the analysis.
- View visualizations, including:
  - Message Trends: A time-based line chart of message counts.
  - Keyword Importance: A bar chart of extracted keywords.
  - Topic Distribution: A pie chart of conversation topics.

Download Report
================
- Click "Generate Report" to create a PDF file summarizing the analysis, including:
  - Top keywords chart.
  - Message trends chart.
  - Summary of top messages.

Key Features Explained
=======================

Chat Parsing
-------------
The app processes chat text files to extract structured data into a DataFrame. It identifies timestamps, senders, and message contents while handling multiline messages and system notifications.

Preprocessing
--------------
Cleans chat messages by:
- Removing stopwords (e.g., "the", "and").
- Tokenizing text.
- Eliminating punctuation.

Topic Modeling
---------------
Utilizes Latent Dirichlet Allocation (LDA) to group conversations into topics. Each topic is represented by a set of key terms.

Visualizations
---------------
Interactive charts provide actionable insights, making it easier to understand chat trends and topics.