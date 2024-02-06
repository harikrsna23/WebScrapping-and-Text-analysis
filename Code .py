#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl


# In[2]:


df = pd.read_excel('Input.xlsx')
urls = df['URL'].tolist()
url_ids = df['URL_ID'].tolist()


# In[3]:


# Creating a directory to store text files (if it doesn't exist)
import os
if not os.path.exists('article_texts'):
    os.makedirs('article_texts')

# Iterating through the list of URLs and their corresponding URL_IDs
for url, url_id in zip(urls, url_ids):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extracting the article title and text
        title = soup.find('title').text.strip()
        article_text = ' '.join([p.text for p in soup.find_all('p')])

        # Saving the extracted data to a text file
        file_name = f'article_texts/{url_id}.txt'
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(f'Title: {title}\n\n')
            file.write(f'Text: {article_text}')

    except Exception as e:
        print(f"Error processing {url}: {str(e)}")


# In[16]:


import nltk
from nltk.corpus import stopwords
import re

# Download NLTK stop words data
nltk.download('stopwords')

# Function to clean the text using stop words
def clean_text_with_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word.lower() not in stop_words]
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text

# Load positive words from positive.txt
with open('positive-words.txt', 'r') as file:
    positive_words = set(file.read().split())

# Load negative words from negative.txt
with open('negative-words.txt', 'r') as file:
    negative_words = set(file.read().split())

def create_positive_negative_word_dictionary(text):
    # Load positive words from positive.txt
    with open('positive-words.txt', 'r') as file:
        positive_words = set(file.read().split())

    # Load negative words from negative.txt
    with open('negative-words.txt', 'r') as file:
        negative_words = set(file.read().split())

    # Initialize sets to collect positive and negative words
    positive_words_in_text = set()
    negative_words_in_text = set()

    for word in text.split():
        if word in positive_words:
            positive_words_in_text.add(word)
        elif word in negative_words:
            negative_words_in_text.add(word)

    return positive_words_in_text, negative_words_in_text


# Function to extract derived variables
def extract_text_variables(text):
    # Clean the text using stop words
    cleaned_text = clean_text_with_stopwords(text)

    # Extract positive and negative words
    positive_words, negative_words = create_positive_negative_word_dictionary(cleaned_text)

    # Calculate Sentimental Analysis variables
    positive_score = sum(1 for word in cleaned_text.split() if word in positive_words)
    negative_score = -sum(1 for word in cleaned_text.split() if word in negative_words)
    total_words = len(cleaned_text.split())
    polarity_score = (positive_score - negative_score) / (positive_score + negative_score + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (total_words + 0.000001)

    # Calculate Analysis of Readability variables
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    avg_sentence_length = len(words) / len(sentences)
    
    complex_word_count = sum(1 for word in words if len(word) > 2)  # Assuming words with more than 2 characters are complex
    percentage_of_complex_words = complex_word_count / total_words
    fog_index = 0.4 * (avg_sentence_length + percentage_of_complex_words)

    # Calculate Average Number of Words Per Sentence
    avg_num_words_per_sentence = total_words / len(sentences)

    # Calculate Word Count
    word_count = len(words)

    # Calculate Syllable Count Per Word
    def syllable_count(word):
        word = word.lower()
        if len(word) <= 3:
            return 1
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if word.endswith("le"):
            count += 1
        if count == 0:
            count = 1
        return count

    syllable_count_per_word = sum(syllable_count(word) for word in words) / total_words

    # Calculate Personal Pronouns
    personal_pronouns = re.findall(r'\b(?:I|we|my|ours|us)\b', cleaned_text, flags=re.IGNORECASE)
    personal_pronoun_count = len(personal_pronouns)

    # Calculate Average Word Length
    avg_word_length = sum(len(word) for word in words) / total_words

    return (positive_score, negative_score, polarity_score, subjectivity_score,
            avg_sentence_length, percentage_of_complex_words, fog_index,
            avg_num_words_per_sentence, complex_word_count, word_count,
            syllable_count_per_word, personal_pronoun_count, avg_word_length)

# Create a list to store the analysis results
analysis_results = []

# Read the Excel file with URLs and URL_IDs
df = pd.read_excel('Input.xlsx')
urls = df['URL'].tolist()
url_ids = df['URL_ID'].tolist()

# Iterate through the list of URL_IDs and perform Text Analysis
for url_id in url_ids:
    file_name = f'article_texts/{url_id}.txt'
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            text = file.read()
            results = extract_text_variables(text)
            
            analysis_result = [url_id, text] + list(results)
            analysis_results.append(analysis_result)
    except Exception as e:
        print(f"Error processing URL_ID {url_id}: {str(e)}")

# Create a DataFrame with the results
column_names = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
                'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT',
                'WORD COUNT', 'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
result_df = pd.DataFrame(analysis_results, columns=column_names)

# Save the DataFrame to an Excel file
result_df.to_excel('Output Data Structure.xlsx', index=False)


# In[ ]:




