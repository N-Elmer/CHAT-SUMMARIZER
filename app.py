# Import Libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import re, string, os, pytz, json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import LdaModel
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (Table, TableStyle, Paragraph, Image, Spacer, SimpleDocTemplate, NextPageTemplate, PageBreak)
from reportlab.lib import styles, enums, colors, pagesizes

# Function Definitions
def save_lda_checkpoint(lda_model, dictionary, checkpoint_dir="Model", checkpoint_name="lda_checkpoint"):
    """Saves the LDA model and dictionary as a checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.model")
    dict_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.dict")
    
    lda_model.save(model_path)
    dictionary.save(dict_path)
    print(f"Checkpoint Saved: {model_path}, {dict_path}")

def load_lda_checkpoint(checkpoint_dir="Model", checkpoint_name="lda_checkpoint"):
    """Loads the LDA model and dictionary from a checkpoint."""
    model_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.model")
    dict_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.dict")
    
    if not os.path.exists(model_path) or not os.path.exists(dict_path):
        raise FileNotFoundError("Checkpoint Files Not Found!")
    
    lda_model = LdaModel.load(model_path)
    dictionary = corpora.Dictionary.load(dict_path)
    print(f"Checkpoint Loaded: {model_path}, {dict_path}")
    return lda_model, dictionary

def process_json_to_txt(file_content):
    """Converts a JSON chat file into text format."""
    try:
        data = json.loads(file_content.decode("utf-8"))
    except json.JSONDecodeError:
        st.error("The uploaded JSON file is not valid.")
        return None

    output_lines = []
    for message in data.get("messages", []):
        if message.get("type") == "message":
            try:
                dt = datetime.fromisoformat(message["date"])
                date_formatted = dt.strftime('%d/%m/%Y, %I:%M %p').lower()
                username = message.get("from", "Unknown")
                content = message.get("text", "")
                line = f"{date_formatted} - {username}: {content}"
                output_lines.append(line)
            except (ValueError, KeyError):
                continue

    return "\n".join(output_lines)

def parse_chat(file_content):
    """Parses chat file content into a DataFrame."""
    chat_data = []
    data = file_content.decode("utf-8").splitlines()
    line_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}\s(?:am|pm)) - (.+?): (.+)")
    system_message_pattern = re.compile(r"(\d{1,2}/\d{1,2}/\d{4}), (\d{1,2}:\d{2}\s(?:am|pm)) - (.+)")

    for line in data:
        if match := line_pattern.match(line):
            chat_data.append([*match.groups()])
        elif sys_match := system_message_pattern.match(line):
            date, time, message = sys_match.groups()
            chat_data.append([date, time, None, message])
        elif chat_data:
            chat_data[-1][3] += f" {line.strip()}"

    return pd.DataFrame(chat_data, columns=['Date', 'Time', 'Sender', 'Message'])

def preprocess_chat(chat_df):
    """Cleans and preprocesses chat messages."""
    chat_df = chat_df[chat_df['Sender'].notnull()].copy()
    def clean_message(message):
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(message.lower())
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        return ' '.join(tokens)
    chat_df['Cleaned_Message'] = chat_df['Message'].apply(clean_message)
    return chat_df

def filter_by_criteria(chat_df, start_date=None, end_date=None, start_time=None, end_time=None, keywords=None):
    """Filters the chat DataFrame based on user-specified criteria."""
    chat_df['Datetime'] = pd.to_datetime(chat_df['Date'] + ' ' + chat_df['Time'], format='%d/%m/%Y %I:%M %p')

    if start_date:
        chat_df = chat_df[chat_df['Datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        chat_df = chat_df[chat_df['Datetime'] <= pd.to_datetime(end_date)]
    if start_time or end_time:
        chat_df = chat_df[(start_time is None or chat_df['Datetime'].dt.time >= start_time) &
                          (end_time is None or chat_df['Datetime'].dt.time <= end_time)]
    if keywords:
        pattern = '|'.join(map(re.escape, filter(None, map(str.strip, keywords))))
        chat_df = chat_df[chat_df['Cleaned_Message'].str.contains(pattern, na=False, case=False)]

    return chat_df

def generate_summary(chat_df, top_n_keywords=10, top_n_messages=5, num_topics=3):
    """Generates a summary of the chat."""
    # Extract Keywords
    vectorizer = TfidfVectorizer(max_features=top_n_keywords)
    tfidf_matrix = vectorizer.fit_transform(chat_df['Cleaned_Message'])
    
    # Create a dictionary of keywords and their scores
    keywords_dict = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0]))
    
    # Summarize Messages
    chat_df['Message_Length'] = chat_df['Message'].str.len()
    summary = chat_df.sort_values(by='Message_Length', ascending=False).head(top_n_messages)
    
    # Extract Topics
    tokenizer = re.compile(r'\w+')
    texts = [tokenizer.findall(msg.lower()) for msg in chat_df['Cleaned_Message']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=5)
    
    return keywords_dict, summary, topics

def plot_topics_distribution(topics):
    """Plot a pie chart of topics distribution."""
    topic_labels = [f"Topic {i+1}" for i, _ in enumerate(topics)]
    topic_words = [', '.join([word.split('*')[1].strip('"') for word in topic.split(' + ')]) for _, topic in topics]
    topic_sizes = [len(words.split(', ')) for words in topic_words]

    # Create a DataFrame for the pie chart
    topics_df = pd.DataFrame({
        'Topic': topic_labels,
        'Words': topic_words,
        'Size': topic_sizes
    })

    # Plot pie chart
    fig = px.pie(
        topics_df,
        names='Topic',
        values='Size',
        title='Distribution',
        hover_data=['Words']
    )
    st.plotly_chart(fig)

def plot_message_trends(chat_df):
    """Plot a line chart of message trends over time."""
    chat_df['Date_Only'] = chat_df['Datetime'].dt.date
    message_counts = chat_df.groupby('Date_Only').size().reset_index(name='Message Count')

    # Plot line chart with area under the curve
    st.line_chart(
        message_counts.set_index('Date_Only'),
        use_container_width=True,
        height=400
    )

def generate_pdf_report(keywords_chart_path, trends_chart_path, top_messages):
    """Generates a PDF report containing chat analysis results with keywords chart on first page and messages table on second page."""
    os.makedirs("Report", exist_ok=True)
    pdf_path = "Report/chat_analysis_report.pdf"
    
    # Use a smaller page size to ensure content fits
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=30,
        leftMargin=30,
        topMargin=30,
        bottomMargin=30
    )
    
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    title_style = styles['Title']
    title_style.spaceAfter = 30
    elements.append(Paragraph("Chat Analysis Report", title_style))

    # Add keywords chart on the first page
    if keywords_chart_path:
        elements.append(Paragraph("Top Keywords", styles['Heading2']))
        img = Image(keywords_chart_path, width=400, height=300)
        elements.append(img)
        elements.append(Spacer(1, 20))

    # Force a page break
    elements.append(PageBreak())

    # Add messages table on the second page
    elements.append(Paragraph("Message Summary", styles['Heading2']))
    elements.append(Spacer(1, 20))
    
    # Create a custom style for table cells
    table_style = styles["BodyText"].clone('TableCell', fontSize=8, leading=10)
    
    # Prepare table data with wrapped text
    table_data = [["Sender", "Message"]]
    for _, row in top_messages.iterrows():
        # Limit message length and wrap text
        sender = Paragraph(str(row["Sender"])[:50], table_style)
        message = Paragraph(str(row["Message"])[:500], table_style)  # Limit message length
        table_data.append([sender, message])
    
    # Create table with adjusted column widths
    available_width = doc.width
    col_widths = [available_width * 0.3, available_width * 0.7]  # 30% for sender, 70% for message
    
    table = Table(
        table_data,
        colWidths=col_widths,
        repeatRows=1  # Repeat header row on each page
    )
    
    # Apply table styles
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWHEIGHT', (0, 0), (-1, -1), None),  # Let row height adjust automatically
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(table)
    
    # Build the PDF
    try:
        doc.build(elements)
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None
        
    return pdf_path

# Generate Summary with LDA Checkpointing
def generate_summary(chat_df, top_n_keywords=10, top_n_messages=5, num_topics=3):
    """
    Generates a summary of the chat with LDA checkpointing.
    """
    # Extract Keywords
    vectorizer = TfidfVectorizer(max_features=top_n_keywords)
    tfidf_matrix = vectorizer.fit_transform(chat_df['Cleaned_Message'])
    keywords_dict = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.sum(axis=0).tolist()[0]))
    
    # Summarize Messages
    chat_df['Message_Length'] = chat_df['Message'].str.len()
    summary = chat_df.sort_values(by='Message_Length', ascending=False).head(top_n_messages)
    
    # Extract Topics
    tokenizer = re.compile(r'\w+')
    texts = [tokenizer.findall(msg.lower()) for msg in chat_df['Cleaned_Message']]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Load or Train LDA Model
    try:
        lda_model, loaded_dictionary = load_lda_checkpoint()
        if loaded_dictionary.token2id != dictionary.token2id:
            print("Dictionary Mismatch; Retraining LDA Model.")
            lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
            save_lda_checkpoint(lda_model, dictionary)
    except FileNotFoundError:
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=10)
        save_lda_checkpoint(lda_model, dictionary)
    
    topics = lda_model.print_topics(num_words=5)
    return keywords_dict, summary, topics

# Streamlit App Structure
st.title("Chat Analyzer")
st.sidebar.header("Upload Chat")

uploaded_file = st.sidebar.file_uploader("Upload A Chat File", type=["txt", "json"])

if uploaded_file:
    # Process and Parse Chat
    if uploaded_file.name.endswith(".json"):
        processed_text = process_json_to_txt(uploaded_file.read())
        if processed_text:
            chat_df = parse_chat(processed_text.encode("utf-8"))
        else:
            st.stop()
    else:
        chat_df = parse_chat(uploaded_file.read())

    st.write("### Raw Chat Data", chat_df)

    with st.spinner("Processing Chat Data..."):
        chat_df = preprocess_chat(chat_df)
        st.success("Chat Data Successfully Preprocessed!")

    # Date and Time Ranges
    chat_df['Datetime'] = pd.to_datetime(chat_df['Date'] + " " + chat_df['Time'], format='%d/%m/%Y %I:%M %p')
    earliest_date, latest_date = chat_df['Datetime'].dt.date.min(), chat_df['Datetime'].dt.date.max()
    current_date, current_time = datetime.now(pytz.timezone("Asia/Dubai")).date(), datetime.now(pytz.timezone("Asia/Dubai")).time()

    # Sidebar Filters and Parameters
    st.sidebar.header("Filters")
    start_date = st.sidebar.date_input("Start Date", value=earliest_date, min_value=earliest_date, max_value=current_date)
    end_date = st.sidebar.date_input("End Date", value=current_date, min_value=earliest_date, max_value=current_date)
    earliest_time = chat_df['Datetime'].dt.time.min()
    latest_time = chat_df['Datetime'].dt.time.max()
    start_time = st.sidebar.time_input("Start Time", value=earliest_time)
    end_time = st.sidebar.time_input("End Time", value=latest_time)
    keywords = st.sidebar.text_input("Keywords [Comma-Separated]").split(",")

    st.sidebar.header("Parameters")
    num_topics = st.sidebar.number_input("Number of Topics", min_value=1, max_value=10, value=3)
    top_n_keywords = st.sidebar.number_input("Top Keywords", min_value=1, max_value=50, value=10)
    top_n_messages = st.sidebar.number_input("Top Messages", min_value=1, max_value=50, value=5)

    filtered_chat = filter_by_criteria(chat_df, start_date, end_date, start_time, end_time, keywords)

    if st.sidebar.button("Apply Filters"):
        keywords, summary, topics = generate_summary(filtered_chat, top_n_keywords, top_n_messages, num_topics)
        st.write("#### Top Keywords")
        st.bar_chart(pd.DataFrame.from_dict(keywords, orient='index', columns=['Score']))
        st.write("#### Topics Distribution")
        plot_topics_distribution(topics)

        st.write("### Filtered Chat Data", filtered_chat)

        if not filtered_chat.empty:
            with st.spinner("Generating Summary..."):
                st.write("#### Message Trends Over Time")
                plot_message_trends(filtered_chat)
                st.write("#### Top Messages")
                st.table(summary[['Sender', 'Message']])
        else:
            st.warning("No Data Available For The Selected Filters.")

    if st.sidebar.button("Generate Report"):
        with st.spinner("Generating PDF Report..."):
            os.makedirs("Report", exist_ok=True)

            # Save Trends Chart
            trends_chart_path = "Report/message_trends_plot.png"

            # Ensure Keywords Chart is Available
            if 'keywords_dict' not in locals():
                keywords_dict, summary, topics = generate_summary(filtered_chat, top_n_keywords, top_n_messages, num_topics)

            # Save Keywords Chart
            keywords_chart_path = "Report/top_keywords.png"
            sns.barplot(x=list(keywords_dict.values()), y=list(keywords_dict.keys()), palette="YlGnBu")
            plt.title("Top Keywords")
            plt.tight_layout()
            plt.savefig(keywords_chart_path)
            plt.close()

            # Generate PDF
            pdf_path = generate_pdf_report(keywords_chart_path, trends_chart_path, summary[['Sender', 'Message']])

            st.success("PDF Report Successfully Generated!")
            # Provide Download Option
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Download Report", pdf_file, file_name="Chat Report.pdf", mime="application/pdf")
