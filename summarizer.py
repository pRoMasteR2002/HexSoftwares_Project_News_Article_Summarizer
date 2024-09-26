import os
import warnings
import logging
import time
import psutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import pandas as pd
import docx
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import random
import torch
from dotenv import load_dotenv

# Update TensorFlow import and warning suppression
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load environment variables
load_dotenv()

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use environment variables for sensitive information
model_name = os.getenv('MODEL_NAME', 'facebook/bart-large-cnn')

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

summarizer = load_model()

def log_resource_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    st.sidebar.text(f"Memory Usage: {memory_info.rss / 1024 / 1024:.2f} MB")

@st.cache_data
def summarize_text(text):
    start_time = time.time()
    summary = summarizer(text, max_length=200, min_length=100, do_sample=False)[0]['summary_text']
    end_time = time.time()
    
    st.sidebar.text(f"Summarization Time: {end_time - start_time:.2f} seconds")
    
    lines = summary.split('.')
    if len(lines) < 7:
        return '. '.join(lines) + '.'
    elif len(lines) > 10:
        return '. '.join(lines[:9]) + '.'
    else:
        return summary

@st.cache_data
def get_word_count(text):
    return len(text.split())

@st.cache_data
def read_file(file):
    if file.type == "text/csv":
        return pd.read_csv(file), "csv"
    elif file.type == "text/plain":
        content = None
        encodings = ['utf-8', 'iso-8859-1', 'windows-1252']
        for encoding in encodings:
            try:
                content = file.getvalue().decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            st.error("Unable to decode the text file. Please check the file encoding.")
            return None, None
        return pd.DataFrame({"content": [content]}), "txt"
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        content = "\n".join([para.text for para in doc.paragraphs])
        return pd.DataFrame({"content": [content]}), "docx"

st.title("News Article Summarizer")

@st.cache_data
def process_articles(df, content_column, indices):
    results = []
    for idx in indices:
        article = df.iloc[idx]
        summary = summarize_text(article[content_column])
        results.append({
            "index": idx,
            "author": article.get('author', 'N/A'),
            "claps": article.get('claps', 'N/A'),
            "summary": summary,
            "word_count": get_word_count(summary)
        })
    return results

option = st.radio("Choose an option:", ["Summarize Text", "Summarize from File"])

if option == "Summarize Text":
    text_input = st.text_area("Enter the text to summarize (must be more than 100 words):")
    if st.button("Summarize"):
        if len(text_input.split()) > 100:
            summary = summarize_text(text_input)
            st.subheader("Summary:")
            st.write(summary)
            st.write(f"Word count: {get_word_count(summary)}")
        else:
            st.error("The input text must be more than 100 words.")

elif option == "Summarize from File":
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "docx"])
    
    if uploaded_file is not None:
        df, file_type = read_file(uploaded_file)
        
        if df is not None:
            if file_type == "csv":
                # Automatically identify the content column or handle missing column
                if 'content' in df.columns:
                    content_column = 'content'
                else:
                    # Fallback: try to infer the content column
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
                        indices = random.sample(range(len(df)), num_articles)
                        results = process_articles(df, content_column, indices)
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
                        valid_indices = [idx for idx in indices if 0 <= idx < len(df)]
                        if len(valid_indices) != len(indices):
                            st.warning(f"Some indices were invalid and will be skipped.")
                        results = process_articles(df, content_column, valid_indices)
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
                    content = df['content'].iloc[0]
                    if len(content.split()) > 100:
                        summary = summarize_text(content)
                        st.subheader("Summary:")
                        st.write(summary)
                        st.write(f"Word count: {get_word_count(summary)}")
                    else:
                        st.error("The input text must be more than 100 words.")

st.sidebar.title("About")
st.sidebar.info("This app summarizes news articles using machine learning. You can input text directly or upload CSV, TXT, or DOCX files containing articles.")

st.sidebar.title("Performance Metrics")
log_resource_usage()
