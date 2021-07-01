import pandas as pd
import numpy as np
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords ##  stopwords dans différentes langues
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def pre_process(text):
    # Function for removing NonAscii characters
    text = "".join(i for i in text if  ord(i)<128)

    # lowercase
    text=text.lower()

    # Stop Words
    text = text.split()
    stops = stopwords.words("english")
    stops.append('story')
    stops.append('br')
    stops.append('b')
    stops.append('en')
    text = [w for w in text if not w in stops]
    text = " ".join(text)
        
    # remove special characters and digits
    #text=re.sub("(\\d|\\W)+"," ",text)
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)

    # remove html
    html_pattern = re.compile('<.*?>')
    text = html_pattern.sub(r'', text)

    return text

def preprocessing(data):
    df = data
    print('------ df into preprocessing ------')
    print(df)
    df['new_tag'] = df['new_tag'].fillna('autre')
    df['text'] = df['description']
    df[['text']] = df[['text']].astype(str)

    df['text'] = df['text'].apply(lambda x:pre_process(x))

    return df