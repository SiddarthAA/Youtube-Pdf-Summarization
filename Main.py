import fitz
import re
import os
import contextlib
import numpy as np
import faiss
import time
import logging

from youtube_transcript_api import YouTubeTranscriptApi

from transformers import pipeline
from sentence_transformers import SentenceTransformer

import google.generativeai as genai
from g4f.client import Client

import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize

import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=RuntimeWarning)
def suppress_warnings():
    warnings.filterwarnings('ignore', message='`resume_download` is deprecated')
suppress_warnings()

class colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def red(text):
    return '\033[91m' + text + '\033[0m'
def cyan(text):
    return '\033[96m' + text + '\033[0m'
def yellow(text):
    return '\033[93m' + text + '\033[0m'
def bold(text):
    return '\033[1m' + text + '\033[0m'
def green(text):
    return '\033[92m' + text + '\033[0m'
def purple(text):
    return '\033[95m' + text + '\033[0m'
def white(text):
    return '\033[97m' + text + '\033[0m'

# Data Extraction From Youtube Video :
# Call Main Function (get_youtube_data)
# Input : Youtube Video URL
# Output : - "Extract_Content.txt" 
#          - "Extract_Transcript.txt"

def write_sentences_to_file(sentences, filename="Extract_Content.txt"):
    try:
        with open(filename, 'w+') as file:
            for sentence in sentences:
                file.write(sentence + "\n")
        print(purple(f"\nSentences successfully written to {filename}"))
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def write_full_transcript_to_file(transcript_data, filename="Extract_Transcript.txt"):
    try:
        with open(filename, 'w+') as file:
            for item in transcript_data:
                start_time = item['start']
                duration = item['duration']
                end_time = start_time + duration
                text = item['text'].replace("\n", " ")
                file.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")
        print(purple(f"Full transcript successfully written to {filename}"))
    except Exception as e:
        print(f"An error occurred while writing to the file: {e}")

def get_youtube_data(video_url):
    video_id = video_url.split("?v=")[1]

    transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
    transcript = transcript_list.find_manually_created_transcript(['en'])
    transcript_data = transcript.fetch()        

    combined_text = " ".join([item['text'].replace("\n", " ") for item in transcript_data])
    sentences = sent_tokenize(combined_text)

    write_sentences_to_file(sentences=sentences)
    write_full_transcript_to_file(transcript_data=transcript_data)

# Data Extraction From PDF File :
# Call Main Function (get_pdf_data)
# Input : PDF File
# Output : - "Extract_Content_Pdf.txt"

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

# Chunking Funuction For Question Answering : 
# Input : None 
# Output : List Of Text Chunks

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

# Chunking Funuction For Summarization : 
# Input : None
# Output : List Of Text Chunks

def Summary_Chunking():
    max_chunk_length=None

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

# Main LLM Summarization Of Text
def max_min(chunk_length):
    if chunk_length <= 100:
        max_length = max(50, int(chunk_length * 0.8)) 
    elif chunk_length <= 500:
        max_length = max(100, int(chunk_length * 0.6)) 
    else:
        max_length = max(200, int(chunk_length * 0.5)) 
    min_length = int(max_length * 0.5)
    
    return max_length, min_length

# short_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
# long_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
# long_alt_summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")

#Model : bart-large-cnn
#Short Summary Based On Given Text Chunks
def short_summary(chunks):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summaries = []
    for chunk in chunks:
        chunk_length = len(chunk.split(" "))
        max_len, min_len = max_min(chunk_length=chunk_length)
        summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=True, top_k=50, top_p=0.95)
        summaries.append(summary[0]['summary_text'])
    return ("\n\n".join(summaries))
    
#Model : distilbart-cnn-12-6
#Long Summary Based On Given Text Chunks
# def long_summary(chunks):
#     summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
#     summaries = []
#     for chunk in chunks:
#         chunk_length = len(chunk.split(" "))
#         max_len, min_len = max_min(chunk_length=chunk_length)
#         summary = summarizer(chunk, max_length=max_len, min_length=min_len, do_sample=False)
#         summaries.append(summary[0]['summary_text'])
#     return ("\n\n".join(summaries))

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

#Model : bart-large-cnn-samsum
#Long Summary Alt 
def long_alt_summary(chunks):
    summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ("\n\n".join(summaries))

# Main Question Answering Of Youtube Video / PDF File
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

#Gemini / ChatGPT Query
def gemini_answer(question, answer):
    try:
        gemini_api = "AIzaSyBNnKC9IwMUhGYbgpvJVDD4vJFfVZSOt5k"
        genai.configure(api_key=gemini_api)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(f"question: {question}, keyword: {answer}\nwrite me an detailed answer to the question including the keyword")
        return response.text
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def gpt_answer(question, answer):
    try:
        client = Client()

        sample_prompt = "Hi"
        sample_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": sample_prompt}]
        )
        
        prompt = f"Generate a detailed answer to the question: {question}, including the keyword {answer}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    
#Misc 
def loading_animation():
    for i in range(1, 11):
        bar = '[' + '■' * i + '.' * (10 - i) + '] ' + str(i * 10) + '%'
        try:
            print('Loading...', bar, end='\r', flush=True)
        except:
            pass
        time.sleep(0.5)

def newline():
    print("\n")

if __name__ == "__main__":
    print(yellow(bold("""
Menu: 
1) Summarize Youtube Video And Ask Questions
2) Summarize PDF File Content And Ask Questions""")))
    option = int(input(green("Enter Option Number : ")))
    if option == 1:
        yt_utl = input(green("\nEnter Youtube Video URL : "))
        get_youtube_data(yt_utl)

        Summary_Chunks = Summary_Chunking()
        print(yellow(bold("\nMenu:\n1)Short Summary\n2)Long Summary")))
        flag_1 = "Yes"
        while flag_1 == "Yes":
            sum_op = (input(green("\nEnter Summary Type : ")))
            if sum_op == "Short":
                summary = short_summary(Summary_Chunks)
                newline()
                loading_animation()
                print(white(bold("Your Youtube Video Summary ")))
                print(cyan(summary))
            elif sum_op == "Long":
                summary = long_summary(Summary_Chunks)
                newline()
                loading_animation()
                print(white(bold("Your Youtube Video Summary ")))
                print(cyan(summary))
            else:
                print(red(bold("\nINVALID OPTION")))
            print
            flag_1 = input(green("\nWant More Summary? (Enter Yes) : "))

        print(white(bold("\n Time To Ask Questions\n")))
        Question_Chunks = QA_Chunking()
        flag_2 = "Yes"
        while flag_2 == "Yes":
            question = green(input("\nEnter Question : "))
            answer_chunk = most_relevant_chunk(question,Question_Chunks)
            keyword = keyword_answer(question,answer_chunk)

            response = gemini_answer(question, keyword)
            if response == None: 
                response = gpt_answer(question, keyword)
            print(bold(purple("\nAnswer")))
            print(bold(cyan(response)))

            flag_2 = input(green("\nMore Questions ? (Enter Yes) : "))