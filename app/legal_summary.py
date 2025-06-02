import re
import torch
import numpy as np
import pandas as pd
from typing import Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Load summarization model
model_name = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Text cleaning utilities
# -----------------------------
def base_clean(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
    text = re.sub(r'page \d+ of \d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarization_task_clean(input_data):
    """
    Clean legal case text for summarization.
    
    Parameters:
    -----------
    input_data : pd.DataFrame, pd.Series, str, or np.ndarray
        - If DataFrame: expects a 'full_case' column containing case text
        - If Series: processes each element in the series
        - If str: a single legal case text
        - If np.ndarray: array of case texts (each element can be str or array of paragraphs)
    
    Returns:
    --------
    pd.DataFrame, pd.Series, str, or list depending on input type
    """
    
    # Helper function to process a single case
    def process_case(case_content):
        all_sents = []
        
        # Handle case where case_content is already an array of paragraphs
        if isinstance(case_content, (list, np.ndarray)):
            paragraphs = case_content
        else:
            # If it's a string, treat it as a single paragraph
            paragraphs = [case_content]
        
        for para in paragraphs:
            cleaned = base_clean(para)
            
            # Split into sentences, filter out empty or boilerplate
            sents = sent_tokenize(cleaned)
            for sent in sents:
                # Remove very short or boilerplate-only sentences
                if len(sent.split()) < 5:
                    continue
                # Skip sentences that are mostly legal citations or numbers
                if re.search(r'^\s*[\d\s\.\(\)]+$', sent):
                    continue
                all_sents.append(sent)
        
        # Rejoin into a multi-sentence paragraph
        return ' '.join(all_sents)
    
    # Handle DataFrame input
    if isinstance(input_data, pd.DataFrame):
        df_clean = input_data.copy()
        cleaned_cases = []
        
        for case in tqdm(df_clean['full_case'], desc="Cleaning cases"):
            cleaned_cases.append(process_case(case))
        
        df_clean['full_case'] = cleaned_cases
        return df_clean
    
    # Handle Series input
    elif isinstance(input_data, pd.Series):
        cleaned_cases = []
        
        for case in tqdm(input_data, desc="Cleaning cases"):
            cleaned_cases.append(process_case(case))
        
        return pd.Series(cleaned_cases, index=input_data.index)
    
    # Handle string input
    elif isinstance(input_data, str):
        return process_case(input_data)
    
    # Handle numpy array input
    elif isinstance(input_data, np.ndarray):
        cleaned_cases = []
        
        for case in tqdm(input_data, desc="Cleaning cases"):
            cleaned_cases.append(process_case(case))
        
        return cleaned_cases
    
    else:
        raise TypeError(f"Input must be a pandas DataFrame, Series, string, or numpy array. Got {type(input_data)}")


def postprocess_legal_summary(text):
    text = text.replace('\xa0', ' ')
    text = text.replace('\\', '')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])([^\s])', r'\1 \2', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = []
    for sentence in sentences:
        if re.search(r'\S+@\S+', sentence): continue
        if re.search(r'(http[s]?://|www\\.|\\w+\\.(com|org|net|gov|edu|info|uk|us))', sentence, re.IGNORECASE): continue
        if re.search(r'(\d{1,4}/){3,}', sentence): continue
        if re.search(r'(\+?\d{1,2}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', sentence): continue
        cleaned_sentences.append(sentence)
    return ' '.join(cleaned_sentences).strip()

# -----------------------------
# Summarization helpers
# -----------------------------
def chunk_text(text, max_words=500):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def summarize_chunk(chunk, max_len=300, min_len=100):
    input_ids = tokenizer.encode(chunk, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=4,
            repetition_penalty=2.0
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def summarize_batch(texts, max_len=300, min_len=100, batch_size=8):
    """
    Summarize multiple texts in batches for improved performance.
    """
    summaries = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize all texts in the batch
        inputs = tokenizer(
            batch_texts,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            # Generate summaries for the entire batch
            outputs = model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=4,
                repetition_penalty=2.0
            )
        
        # Decode all summaries in the batch
        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        summaries.extend(batch_summaries)
    
    return summaries

def summarize_long_text(text, 
                       chunk_size=1000, 
                       chunk_summary_maxlen=300, 
                       chunk_summary_minlen=100, 
                       final_summary_maxlen=300, 
                       final_summary_minlen=100,
                       drop_last_chunks=0):
    # Create chunks
    chunks = chunk_text(text, max_words=chunk_size)
    
    # Drop last chunks if specified
    if drop_last_chunks > 0:
        if drop_last_chunks >= len(chunks):
            chunks_to_use = chunks
        else:
            chunks_to_use = chunks[:-drop_last_chunks]
    else:
        chunks_to_use = chunks
    
    # Summarize each chunk
    chunk_summaries = []
    for i, chunk in enumerate(chunks_to_use):
        summary = summarize_chunk(chunk, max_len=chunk_summary_maxlen, min_len=chunk_summary_minlen)
        chunk_summaries.append(summary)
    
    # Join chunk-level summaries
    joined_summary = " ".join(chunk_summaries)
    final_summary = summarize_chunk(
        joined_summary,
        max_len=final_summary_maxlen,
        min_len=final_summary_minlen
    )
    
    return final_summary

# -----------------------------
# Interface function
# -----------------------------
def legal_summary_pipeline(input_data: Union[str, np.ndarray, pd.DataFrame], 
                           column_name: str = "full_case", 
                           output_column: str = "summary", 
                           summary_style: str = "long",
                           batch_size: int = 8,
                           use_batch_processing: bool = True):
    """
    Process legal case summaries with optional batch processing.
    
    Parameters:
    -----------
    input_data : Union[str, np.ndarray, pd.DataFrame]
        Input data to summarize
    column_name : str
        Column name containing the case text (for DataFrame input)
    output_column : str
        Column name for the output summaries (for DataFrame input)
    summary_style : str
        One of 'long', 'short', or 'tiny'
    batch_size : int
        Number of texts to process in each batch (for DataFrame input)
    use_batch_processing : bool
        Whether to use batch processing for DataFrames
    """

    settings = {
        "long":   {"chunk_summary_maxlen": 256, "chunk_summary_minlen": 75, "final_summary_maxlen": 600, "final_summary_minlen": 350},
        "short":  {"chunk_summary_maxlen": 100, "chunk_summary_minlen": 50, "final_summary_maxlen": 250, "final_summary_minlen": 100},
        "tiny":   {"chunk_summary_maxlen": 40,  "chunk_summary_minlen": 10, "final_summary_maxlen": 75,  "final_summary_minlen": 25},
    }

    if summary_style not in settings:
        raise ValueError(f"summary_style must be one of {list(settings.keys())}")

    params = settings[summary_style]

    def process_cleaned_text(text):
        summary = summarize_long_text(text, **params)
        return postprocess_legal_summary(summary)
    
    def process_cleaned_text_batch(texts):
        """Process multiple texts in batches."""
        # First, summarize all texts
        summaries = []
        
        # Process each text to get chunks
        all_chunks = []
        text_chunk_counts = []
        
        for text in texts:
            chunks = chunk_text(text, max_words=1000)
            all_chunks.extend(chunks)
            text_chunk_counts.append(len(chunks))
        
        # Summarize all chunks in batches
        chunk_summaries = summarize_batch(
            all_chunks, 
            max_len=params["chunk_summary_maxlen"], 
            min_len=params["chunk_summary_minlen"],
            batch_size=batch_size
        )
        
        # Reconstruct summaries for each text
        chunk_idx = 0
        for count in text_chunk_counts:
            text_chunk_summaries = chunk_summaries[chunk_idx:chunk_idx + count]
            joined_summary = " ".join(text_chunk_summaries)
            
            # Final summarization
            final_summary = summarize_chunk(
                joined_summary,
                max_len=params["final_summary_maxlen"],
                min_len=params["final_summary_minlen"]
            )
            summaries.append(postprocess_legal_summary(final_summary))
            chunk_idx += count
        
        return summaries

    # Single string input
    if isinstance(input_data, str):
        cleaned_texts = summarization_task_clean(input_data)
        return process_cleaned_text(cleaned_texts)

    # Numpy array of paragraph lists
    elif isinstance(input_data, np.ndarray):
        cleaned_texts = summarization_task_clean(input_data)
        return process_cleaned_text(cleaned_texts)

    # DataFrame input
    elif isinstance(input_data, pd.DataFrame):
        df_copy = input_data.copy()
        
        # Clean the data (this will show progress bar)
        df_copy[column_name] = summarization_task_clean(df_copy[column_name])
        
        if use_batch_processing and len(df_copy) > 1:
            # Process in batches for better performance
            all_summaries = []
            cleaned_texts = df_copy[column_name].tolist()
            
            # Process texts in batches with progress bar
            for i in tqdm(range(0, len(cleaned_texts), batch_size), 
                         desc=f"Summarizing cases (batch size: {batch_size})"):
                batch_texts = cleaned_texts[i:i + batch_size]
                batch_summaries = process_cleaned_text_batch(batch_texts)
                all_summaries.extend(batch_summaries)
            
            df_copy[output_column] = all_summaries
        else:
            # Apply summarization with progress bar (single processing)
            tqdm.pandas(desc="Summarizing cases")
            df_copy[output_column] = df_copy[column_name].progress_apply(process_cleaned_text)
        
        return df_copy

    else:
        raise TypeError("Input must be a string, numpy.ndarray, or pandas.DataFrame")