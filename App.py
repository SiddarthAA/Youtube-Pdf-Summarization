import fitz
import re
import os
import numpy as np
import faiss
import time
import streamlit as st
import warnings
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import google.generativeai as genai
from g4f.client import Client

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Functions for data extraction and processing
def write_sentences_to_file(sentences, filename="Extract_Content.txt"):
    with open(filename, 'w+') as file:
        for sentence in sentences:
            file.write(sentence + "\n")

def write_full_transcript_to_file(transcript_data, filename="Extract_Transcript.txt"):
    with open(filename, 'w+') as file:
        for item in transcript_data:
            start_time = item['start']
            duration = item['duration']
            end_time = start_time + duration
            text = item['text'].replace("\n", " ")
            file.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")

def get_youtube_data(video_url):
    video_id = video_url.split("?v=")[1]
    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_manually_created_transcript(['en'])
    transcript_data = transcript.fetch()
    combined_text = " ".join([item['text'].replace("\n", " ") for item in transcript_data])
    sentences = sent_tokenize(combined_text)
    write_sentences_to_file(sentences=sentences)
    write_full_transcript_to_file(transcript_data=transcript_data)

def get_pdf_data(pdf_path="Paper.pdf", output_file_path="Extracted_Content_Pdf.txt"):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        page_text = page.get_text()
        page_text = re.sub(r'\n+', ' ', page_text)
        page_text = re.sub(r'\.{2,}', '', page_text)
        text += page_text
    pdf_document.close()
    try:
        tmp = text.split(" ")
        indx = tmp.index("References")
        text = " ".join(tmp[:indx])
    except: 
        pass
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(text)

def QA_Chunking(max_chunk_length=50):
    with open("Extract_Content.txt","r",encoding="utf-8") as fh: 
        content = fh.read()
    sentences = sent_tokenize(content)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_chunk_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def Summary_Chunking():
    max_chunk_length = None
    with open("Extract_Content.txt","r",encoding="utf-8", errors='ignore') as fh: 
        content = fh.read()
        wrds = len(content.split(" "))
        if wrds <= 1000:
            max_chunk_length = 200
        elif wrds <= 4000: 
            max_chunk_length = 500
        else : 
            max_chunk_length = 1000
    sentences = sent_tokenize(content)
    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_chunk_length:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def max_min(chunk_length):
    if chunk_length <= 100:
        max_length = max(50, int(chunk_length * 0.8)) 
    elif chunk_length <= 500:
        max_length = max(100, int(chunk_length * 0.6)) 
    else:
        max_length = max(200, int(chunk_length * 0.5)) 
    min_length = int(max_length * 0.5)
    return max_length, min_length

def short_summary(chunks):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for chunk in chunks:
        chunk_length = len(chunk.split(" "))
        max_len, min_len = max_min(chunk_length=chunk_length)
        summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=True, top_k=50, top_p=0.95)
        summaries.append(summary[0]['summary_text'])
    return ("\n\n".join(summaries))

def long_summary(chunks):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summaries = []
    for chunk in chunks:
        chunk_length = len(chunk.split(" "))
        if chunk_length <= 100:
            max_len = max(100, int(chunk_length * 1.2))
        elif chunk_length <= 500:
            max_len = max(200, int(chunk_length * 0.8))
        else:
            max_len = max(400, int(chunk_length * 0.6))
        min_len = int(max_len * 0.5)
        input_len = len(chunk)
        if max_len > input_len:
            max_len = max(1, int(input_len * 0.7))
        summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ("\n\n".join(summaries))

def long_alt_summary(chunks):
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ("\n\n".join(summaries))

def get_vector_chunk_single(model_name, text_chunks, query):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    query_embedding = model.encode([query])[0]
    query_embedding = query_embedding / np.linalg.norm(query_embedding)
    D, I = index.search(query_embedding.reshape(1, -1), k=2)
    most_relevant_chunk = text_chunks[I[0][0]]
    return(f"{most_relevant_chunk}")

def most_relevant_chunk(query, text_chunks):
    models = ["all-mpnet-base-v2","distilbert-base-nli-mean-tokens"]
    output_chunks = list()
    for model in models:
        output_chunks.append(get_vector_chunk_single(model, text_chunks, query=query))
    if output_chunks[0]==output_chunks[1]:
        return(output_chunks[0])
    else:
        return(output_chunks[0]+"\n"+output_chunks[1])

def keyword_answer(question, chunk):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-large-squad2", tokenizer="deepset/roberta-large-squad2", model_kwargs={"max_length": 200})
    result = qa_pipeline(question=question, context=chunk)
    return(result['answer'])

def gemini_answer(question, answer):
    try:
        gemini_api = "AIzaSyBNnKC9IwMUhGYbgpvJVDD4vJFfVZSOt5k"
        genai.configure(api_key=gemini_api)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"question: {question}, keyword: {answer}\nwrite me an detailed answer to the question including the keyword")
        return response.text
    except Exception as e:
        return None

def gpt_answer(question, answer):
    try:
        client = Client()
        sample_prompt = "Hi"
        sample_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": sample_prompt}],
        )
        if "choices" in sample_response and len(sample_response["choices"]) > 0:
            prompt = f"question: {question}, keyword: {answer}\nwrite me a detailed answer to the question including the keyword"
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message["content"]
    except Exception as e:
        return None

def bert_answer(question, chunk):
    qa_pipeline = pipeline("question-answering", model="deepset/bert-large-uncased-whole-word-masking-squad2", tokenizer="deepset/bert-large-uncased-whole-word-masking-squad2", model_kwargs={"max_length": 200})
    result = qa_pipeline(question=question, context=chunk)
    return(result['answer'])

# Streamlit app interface
st.title("Extractive and Summarized Content from PDFs and YouTube Videos")

st.sidebar.title("Select Source")
source = st.sidebar.radio("Choose the source of content:", ("PDF", "YouTube Video"))

if source == "PDF":
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        pdf_path = pdf_file.name
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        st.success("PDF file uploaded successfully.")
        get_pdf_data(pdf_path=pdf_path)
        st.info("Extracted content from PDF is saved in 'Extracted_Content_Pdf.txt'")

elif source == "YouTube Video":
    video_url = st.text_input("Enter the YouTube video URL")
    if video_url:
        get_youtube_data(video_url=video_url)
        st.success("YouTube video transcript extracted successfully.")
        st.info("Extracted content from YouTube video is saved in 'Extract_Transcript.txt'")

st.sidebar.title("Select Action")
action = st.sidebar.radio("Choose an action:", ("Summarize Content", "QA Chunking"))

if action == "Summarize Content":
    chunking_method = st.sidebar.selectbox("Choose a chunking method:", ("Short Summary", "Long Summary", "Alternative Long Summary"))
    chunks = Summary_Chunking()
    if chunking_method == "Short Summary":
        summary = short_summary(chunks)
    elif chunking_method == "Long Summary":
        summary = long_summary(chunks)
    else:
        summary = long_alt_summary(chunks)
    st.subheader("Summarized Content")
    st.write(summary)

elif action == "QA Chunking":
    query = st.text_input("Enter your query for QA:")
    if query:
        chunks = QA_Chunking()
        most_relevant = most_relevant_chunk(query, chunks)
        st.subheader("Most Relevant Chunk")
        st.write(most_relevant)
        keyword = keyword_answer(query, most_relevant)
        st.subheader("Keyword Answer")
        st.write(keyword)
        gemini = gemini_answer(query, keyword)
        st.subheader("Gemini Answer")
        st.write(gemini)
        gpt = gpt_answer(query, keyword)
        st.subheader("GPT Answer")
        st.write(gpt)
        bert = bert_answer(query, most_relevant)
        st.subheader("BERT Answer")
        st.write(bert)