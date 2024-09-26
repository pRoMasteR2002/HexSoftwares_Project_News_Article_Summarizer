
#How to run:
# 1. Install the required libraries:
#    pip install psutil streamlit pandas python-docx transformers torch python-dotenv tensorflow (for all together)

#    OR

#    For installing the libraries separately, run the following commands:
#    pip install streamlit
#    pip install pandas
#    pip install python-docx
#    pip install transformers
#    pip install torch
#    pip install python-dotenv
#    pip install tensorflow
#    pip install psutil

#2. Run the application via Streamlit by using the command:
#   python -m streamlit run <script_name>.py
#   Replace <script_name> with the name of this file. For example, if the file is named summarizer.py, run:
#   (python -m streamlit run summarizer.py)

#3. You will automatically redirected to the web app in the browser to input text or upload files for summarization.


import os  # Provides operating system-related functions
import warnings  # Handles warnings
import logging  # For logging system messages and errors
import time  # To measure time for performance tracking
import psutil  # To track system resource usage
import streamlit as st  # For building the web application interface
import pandas as pd  # For handling CSV and other tabular data
import docx  # For reading DOCX files
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM  # For loading the summarization model
import random  # For selecting random articles for summarization
import torch  # For GPU support and tensor computations
from dotenv import load_dotenv  # For loading environment variables

# Suppress TensorFlow and other warnings to reduce clutter in logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# TensorFlow setup for compatibility and warning suppression
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load environment variables from a .env file (for sensitive information like model names)
load_dotenv()

# Check if a GPU (CUDA) is available, otherwise fall back to CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Fetch the model name from the environment, or use a default BART model
model_name = os.getenv('MODEL_NAME', 'facebook/bart-large-cnn')

@st.cache_resource  # Cache the model to avoid reloading every time
def load_model():
    """
    Load the pre-trained BART model and tokenizer for summarization from Hugging Face.
    The model is loaded on the GPU if available, otherwise on CPU.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # Load tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)  # Load model
    # Create summarization pipeline
    return pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

# Load the model and tokenizer
summarizer = load_model()

def log_resource_usage():
    """
    Logs memory usage of the process and displays it in the Streamlit sidebar.
    """
    process = psutil.Process()  # Get the current process
    memory_info = process.memory_info()  # Get memory usage
    st.sidebar.text(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")  # Display in MB

@st.cache_data  # Cache function to avoid recomputation unless input changes
def summarize_text(text):
    """
    Summarizes the given text using the BART model.
    
    Parameters:
    - text (str): The input text to summarize.

    Returns:
    - summary (str): The summarized text.
    """
    start_time = time.time()  # Start time to measure performance
    summary = summarizer(text, max_length=200, min_length=100, do_sample=False)[0]['summary_text']  # Summarize text
    end_time = time.time()  # End time after summarization

    # Log time taken for summarization
    st.sidebar.text(f"Summarization Time: {end_time - start_time:.2f} seconds")

    # Ensure summary has a reasonable number of sentences (7-10 sentences)
    lines = summary.split('.')
    if len(lines) < 7:
        return '. '.join(lines) + '.'
    elif len(lines) > 10:
        return '. '.join(lines[:9]) + '.'
    else:
        return summary

@st.cache_data  # Cache function to avoid recomputation unless input changes
def get_word_count(text):
    """
    Returns the word count of the input text.
    
    Parameters:
    - text (str): The input text.
    
    Returns:
    - (int): The word count.
    """
    return len(text.split())

@st.cache_data  # Cache function to avoid recomputation unless input changes
def read_file(file):
    """
    Reads the uploaded file and converts it to a pandas DataFrame.
    
    Parameters:
    - file (UploadedFile): The uploaded file from Streamlit.
    
    Returns:
    - df (DataFrame): DataFrame containing the text content.
    - file_type (str): Type of file (csv, txt, docx).
    """
    if file.type == "text/csv":
        return pd.read_csv(file), "csv"  # Read CSV files
    elif file.type == "text/plain":
        # Handle plain text files with multiple encodings
        content = None
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                content = file.getvalue().decode(encoding)  # Decode file content
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            st.error("Unable to decode the text file. Please check the file encoding.")
            return None, None
        return pd.DataFrame({"content": [content]}), "txt"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Handle DOCX files
        doc = docx.Document(file)
        content = "\n".join([para.text for para in doc.paragraphs])
        return pd.DataFrame({"content": [content]}), "docx"

# Streamlit app title
st.title("News Article Summarizer")

@st.cache_data  # Cache function to avoid recomputation unless input changes
def process_articles(df, content_column, indices):
    """
    Processes and summarizes articles from a DataFrame.
    
    Parameters:
    - df (DataFrame): DataFrame containing the articles.
    - content_column (str): The column name where the text is stored.
    - indices (list): List of row indices to summarize.
    
    Returns:
    - results (list of dict): List of dictionaries containing the article summary and metadata.
    """
    results = []
    for idx in indices:
        article = df.iloc[idx]  # Get the article by index
        summary = summarize_text(article[content_column])  # Summarize the article text
        results.append({
            "index": idx,
            "author": article.get('author', 'N/A'),  # Get author if available
            "claps": article.get('claps', 'N/A'),  # Get claps if available (e.g., for Medium articles)
            "summary": summary,
            "word_count": get_word_count(summary)  # Get word count of the summary
        })
    return results

# Main user interface for text or file summarization
option = st.radio("Choose an option:", ["Summarize Text", "Summarize from File"])

if option == "Summarize Text":
    # Text summarization option
    text_input = st.text_area("Enter the text to summarize (must be more than 100 words):")
    if st.button("Summarize"):
        if len(text_input.split()) > 100:
            summary = summarize_text(text_input)  # Summarize input text
            st.subheader("Summary:")
            st.write(summary)
            st.write(f"Word count: {get_word_count(summary)}")
        else:
            st.error("The input text must be more than 100 words.")

elif option == "Summarize from File":
    # File summarization option
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "docx"])
    
    if uploaded_file is not None:
        df, file_type = read_file(uploaded_file)  # Read the uploaded file
        
        if df is not None:
            if file_type == "csv":
                # Automatically identify the content column or handle missing column
                if 'content' in df.columns:
                    content_column = 'content'
                else:
                    # Infer the content column from string columns
                    possible_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str)).all()]
                    if possible_columns:
                        content_column = possible_columns[0]  # Use the first matching column
                    else:
                        st.error("No suitable text column found in the CSV file.")
                        st.stop()
                
                summarize_option = st.radio("Choose summarization option:", ["Random Articles", "Specific Articles"])
                
                if summarize_option == "Random Articles":
                    num_articles = st.number_input("Enter the number of articles to summarize:", min_value=1, max_value=len(df), value=1)
                    if st.button("Summarize Random Articles"):
                        indices = random.sample(range(len(df)), num_articles)  # Randomly select article indices
                        results = process_articles(df, content_column, indices)  # Summarize the selected articles
                        for result in results:
                            st.subheader(f"Article {result['index']}")
                            st.write(f"Author: {result['author']}")
                            st.write(f"Claps: {result['claps']}")
                            st.write("Summary:")
                            st.write(result['summary'])
                            st.write(f"Word count: {result['word_count']}")
                            st.write("---")
                
                elif summarize_option == "Specific Articles":
                    article_indices = st.text_input("Enter the indices of articles to summarize (comma-separated):")
                    if st.button("Summarize Specific Articles"):
                        indices = [int(idx.strip()) for idx in article_indices.split(",") if idx.strip().isdigit()]
                        valid_indices = [idx for idx in indices if 0 <= idx < len(df)]  # Ensure valid indices
                        if len(valid_indices) != len(indices):
                            st.warning(f"Some indices were invalid and will be skipped.")
                        results = process_articles(df, content_column, valid_indices)  # Summarize specified articles
                        for result in results:
                            st.subheader(f"Article {result['index']}")
                            st.write(f"Author: {result['author']}")
                            st.write(f"Claps: {result['claps']}")
                            st.write("Summary:")
                            st.write(result['summary'])
                            st.write(f"Word count: {result['word_count']}")
                            st.write("---")
            else:  # For txt and docx files
                if st.button("Summarize"):
                    content = df['content'].iloc[0]  # Get the first document's content
                    if len(content.split()) > 100:
                        summary = summarize_text(content)  # Summarize content
                        st.subheader("Summary:")
                        st.write(summary)
                        st.write(f"Word count: {get_word_count(summary)}")
                    else:
                        st.error("The input text must be more than 100 words.")

# Sidebar information
st.sidebar.title("About")
st.sidebar.info("This app summarizes news articles using machine learning. You can input text directly or upload CSV, TXT, or DOCX files containing articles.")

# Performance metrics displayed in the sidebar
st.sidebar.title("Performance Metrics")
log_resource_usage()
