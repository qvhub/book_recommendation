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

def recommend_title(df, title, genre, leng, sel_book=0, author=True):
    # title ==> title of book
    # genre ==> category of book
    # leng ==> nbr of book
    # sel_book ==> phase of recommendation
    # authors ==> author of book

    print('ok recommendation title')
    print(df)

    #####
    # Matching the genre with the dataset and reset the index
    data = df.loc[df['new_tag'] == genre]
    data.reset_index(level = 0, inplace = True) 
    

    # Convert the index into series
    indices = pd.Series(data.index, index = data['title'])
    print('ok indices')
    
    # name of authors
    var_author = data['authors'][indices[title]]
    print(var_author)
    print('ok author')

    # dataset with selected author
    if author == True:
        dt = data.loc[data['authors'] == var_author]
        ### secu si <=2 lignes ##########
        if len(dt) <= 2:
            dt = data
            # recommend 1
            # sel_book==0 ==> 2 books
            if sel_book == 0:
                sel = 1
            # recommend 2
            # sel_book==4 ==> 3 books
            elif sel_book == 4:
                sel = 0
            # recommend 3
            # sel_book==7 ==> 1 books
            elif sel_book == 7:
                sel = 0
            # recommend 3
            # sel_book==11 ==> 1 books
            elif sel_book == 11:
                sel = 0
            # recommend 4
            # sel_book==13 ==> 1 books
            elif sel_book == 13:
                sel = 0
            # recommend 4
            # sel_book==15 ==> 1 books
            elif sel_book == 15:
                sel = 0
            # recommend 4
            # sel_book==17 ==> 1 books
            elif sel_book == 17:
                sel = 0
        else:
            # recommend 1
            # sel_book==0 ==> 2 books
            if sel_book == 0:
                sel = 1
            # recommend 2
            # sel_book==4 ==> 3 books
            elif sel_book == 4:
                sel = 0
            # recommend 3
            # sel_book==7 ==> 3 books
            elif sel_book == 7:
                sel = 0
            # recommend 3
            # sel_book==11 ==> 1 books
            elif sel_book == 11:
                sel = 0
            # recommend 4
            # sel_book==13 ==> 1 books
            elif sel_book == 13:
                sel = 0
            # recommend 4
            # sel_book==15 ==> 1 books
            elif sel_book == 15:
                sel = 0
            # recommend 4
            # sel_book==17 ==> 1 books
            elif sel_book == 17:
                sel = 0
    # dataset without selected author   
    elif author == False :
        if len(data.loc[data['authors'] == var_author]) <=2:
            dt = data
            # recommend 1
            # sel_book==1 ==> 1 book
            if sel_book == 1:
                sel = 3
            # recommend 3
            # sel_book==9 ==> 1 book
            elif sel_book == 9:
                sel = 0
        else:
            # concate dataframe book reference with select. without author
            df_ref_book = data[data['title'] == title]
            dt = data.loc[data['authors'] != var_author]
            dt = pd.concat([dt, df_ref_book])
            # recommend 1
            # sel_book==1 ==> 1 book
            if sel_book == 1:
                sel = 1
            # recommend 3
            # sel_book==9 ==> 1 book
            elif sel_book == 9:
                sel = 0
    
    #Converting the book title into vectors and used bigram
    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1, stop_words='english')
    tfidf_matrix = tf.fit_transform(dt['title'])
    
    # Calculating the similarity measures based on Cosine Similarity
    sg = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index corresponding to original_title
    idx = indices[title]
    
    dt.reset_index(level = 0, inplace = True)
    idx = dt[dt['level_0']==idx].index.tolist()[0]
    
    #display(dt)

    # Get the pairwsie similarity scores 
    sig = list(enumerate(sg[idx]))

    # Sort the books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)

    # Scores of the nbr == leng most similar books 
    sig = sig[sel:leng+sel]

    # Book indicies
    movie_indices = [i[0] for i in sig]
    movie_indices = list([dt['index'].iloc[i] for i in movie_indices])

    # Top nbr == leng book recommendation
    rec = df[['title', 'small_image_url', 'authors', 'average_rating']].iloc[movie_indices]
       
    # It reads the top nbr == leng recommend book url and print the images
    '''
    for i in rec['small_image_url']:
        response = requests.get(i)
        img = Image.open(BytesIO(response.content))
        plt.figure()
        print(plt.imshow(img))
    '''
    return rec

def recommend_descrip(df, title, genre, leng, sel_book, author=True):
    # title ==> title of book
    # genre ==> category of book
    # leng ==> nbr of book
    # sel_book ==> phase of recommendation
    # authors ==> author of book

    #####
    # Matching the genre with the dataset and reset the index
    data = df.loc[df['new_tag'] == genre]  
    data.reset_index(level = 0, inplace = True) 
  
    # Convert the index into series
    indices = pd.Series(data.index, index = data['title'])
    
    # name of authors
    var_author = data['authors'][indices[title]]
    print(var_author)
    # dataset with or without selected author
    if author == True:
        dt = data.loc[data['authors'] == var_author]
        
        ### secu si 1 ligne ##########
        if len(dt) <= 3:
            dt = data
            # recommend n°1
            # sel_book==2 ==> 3 books
            if sel_book == 2:
                sel = 1
            # recommend n°2
            # sel_book==5 ==> 3 books
            elif sel_book == 5:
                sel = 1
            # recommend n°3
            # sel_book==8 ==> 2 books
            elif sel_book == 8:
                sel = 1
            # recommend n°3
            # sel_book==12 ==> 2 books
            elif sel_book == 12:
                sel = 1
            # recommend n°4
            # sel_book==14 ==> 2 books
            elif sel_book == 14:
                sel = 1
            # recommend n°
            # sel_book==16 ==> 2 books
            elif sel_book == 16:
                sel = 1
            # recommend n°4
            # sel_book==18 ==> 2 books
            elif sel_book == 18:
                sel = 1
        else:
            # recommend n°1
            # sel_book==1 ==> 3 books
            if sel_book == 2:
                sel = 1
            # recommend n°2
            # sel_book==5 ==> 3 books
            elif sel_book == 5:
                sel = 1
            # recommend n°3
            # sel_book==8 ==> 1 books
            elif sel_book == 8:
                sel = 1
            # recommend n°3
            # sel_book==12 ==> 2 books
            elif sel_book == 12:
                sel = 1
            # recommend n°4
            # sel_book==14 ==> 2 books
            elif sel_book == 14:
                sel = 1
            # recommend n°
            # sel_book==16 ==> 2 books
            elif sel_book == 16:
                sel = 1
            # recommend n°4
            # sel_book==18 ==> 2 books
            elif sel_book == 18:
                sel = 1
    elif author == False:
        if len(data.loc[data['authors'] == var_author]) <=3:
            dt = data
            # recommend n°1
            # sel_book==3 ==> 3 books
            if sel_book == 3:
                sel = 4
            # recommend n°2
            # sel_book==6 ==> 3 books
            elif sel_book == 6:
                sel = 4
            # recommend n°3
            # sel_book==10 ==> 2 books
            elif sel_book == 10:
                sel = 1
        else:
            df_ref_book = data[data['title'] == title]
            dt = data.loc[data['authors'] != var_author]
            dt = pd.concat([dt, df_ref_book])
            # recommend n°1
            # sel_book==3 ==> 3 books
            if sel_book == 3:
                sel = 1
            # recommend n°2
            # sel_book==6 ==> 3 books
            elif sel_book == 6:
                sel = 1
            # recommend n°3
            # sel_book==10 ==> 2 books
            elif sel_book == 10:
                sel = 1
    #Converting the book description into vectors and used bigram
    tf = TfidfVectorizer(analyzer='word', ngram_range=(2, 2), min_df = 1, stop_words='english')
    tfidf_matrix = tf.fit_transform(dt['text'])
    
    # Calculating the similarity measures based on Cosine Similarity
    sg = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index corresponding to original_title
    idx = indices[title]
    dt.reset_index(level = 0, inplace = True)
    idx = dt[dt['level_0']==idx].index.tolist()[0]
        
    # Get the pairwsie similarity scores 
    sig = list(enumerate(sg[idx]))
    
    # Sort the books
    sig = sorted(sig, key=lambda x: x[1], reverse=True)

    # Scores of the nbr == leng most similar books
    sig = sig[sel:leng+sel]

    # Book indicies
    movie_indices = [i[0] for i in sig]
    movie_indices = list([dt['index'].iloc[i] for i in movie_indices])

    # Top nbr == leng book recommendation
    rec = df[['title', 'small_image_url', 'authors', 'average_rating']].iloc[movie_indices]
    '''
    # It reads the top nbr == leng recommend book url and print the images
    for i in rec['small_image_url']:
        response = requests.get(i)
        img = Image.open(BytesIO(response.content))
        plt.figure()
        print(plt.imshow(img))
    #display(rec)
    '''
    return rec


def content_base_1(df, title):
    # df ==> dataframe to use
    # title ==> title of book

    ##### dataframe to use
    data = df
    
    # Convert the index into series
    indices = pd.Series(data.index, index = data['title'])
    # name of genre
    var_genre = data['new_tag'][indices[title]]
    
    # Top 2 book recommendation title with a same author
    rec_title_with_author = recommend_title(data, title, var_genre, 2, 0, True)
    
    # Top 1 book recommendation title with other authors
    rec_title_without_author = recommend_title(data, title, var_genre, 1, 1, False)
    
    # Top 3 book recommendation description with a same author
    rec_desc_with_author = recommend_descrip(data, title, var_genre, 3, 2, True)
    
    # Top 3 book recommendation description with other authors

    rec_desc_without_author = recommend_descrip(data, title, var_genre, 3, 3, False)
    
    rec_content_1 = pd.concat([rec_title_with_author, rec_title_without_author, rec_desc_with_author, rec_desc_without_author])
    #rec_content_1 = pd.concat([rec_title, rec_desc_with_author])

    #display(rec_content_1)
    dic_book = rec_content_1.to_dict('records')
    return dic_book

def content_base_2(df, cat1):
    # df ==> dataframe to use
    # cat1 ==> selection category


    #### dataframe to use
    df = df
    print('ok data')

    data = df.loc[df['new_tag'] == cat1].sort_values(by = 'average_rating', ascending = False)
    data.reset_index(level = 0, inplace = True)

    # name of title
    title = data['title'][0]
    print(title)
    print('ok title')

    # Top 3 book recommendation title
    rec_title = recommend_title(df, title, cat1, 3, 4, True)
    print('--- rec title')
    print(rec_title)
    
    # Top 3 book recommendation description with a same author
    rec_desc_with_author = recommend_descrip(df, title, cat1, 3, 5, True)
    print('--- rec_desc_with_author')
    print(rec_title)

    # Top 3 book recommendation description with other authors
    rec_desc_without_author = recommend_descrip(df, title, cat1, 3, 6, False)
    print('--- rec_desc_without_author')
    print(rec_desc_without_author)

    rec_content_1 = pd.concat([rec_title, rec_desc_with_author, rec_desc_without_author])
    rec_content_1 = pd.concat([rec_title, rec_desc_with_author])
    dic_book = rec_content_1.to_dict('records')
    return dic_book

def content_base_3(df, cat1, cat2):
    # df ==> dataframe to use
    # cat1 ==> selection category


    #### dataframe to use
    data = df

    #### dataframe to use for cat1
    data_cat1 = data.loc[df['new_tag'] == cat1].sort_values(by = 'average_rating', ascending = False)
    data_cat1.reset_index(level = 0, inplace = True)

    #### dataframe to use for cat2
    data_cat2 = data.loc[df['new_tag'] == cat2].sort_values(by = 'average_rating', ascending = False)
    data_cat2.reset_index(level = 0, inplace = True)

    # name of title cat 1 first book
    title_1 = data_cat1['title'][0]
    print(title_1)

    # name of title cat 1 second book
    title_2 = data_cat1['title'][1]
    print(title_2)

    # name of title cat 2 
    title_3 = data_cat2['title'][0]
    print(title_3)

    # Top 1 book recommendation title cat 1
    rec_title_cat1_bk1 = recommend_title(data, title_1, cat1, 1, 7, True)

    # Top 1 book recommendation title cat 1
    rec_desc_cat1_bk1 = recommend_descrip(data, title_1, cat1, 2, 8, True)

    # Top 2 book recommendation title cat 1
    rec_title_cat1_bk2 = recommend_title(data, title_2, cat1, 1, 9, False)

    # Top 2 book recommendation title cat 1
    rec_desc_cat1_bk2 = recommend_descrip(data, title_2, cat1, 2, 10, False)
    
    # Top 1 book recommendation title cat 2
    rec_title_cat2_bk1 = recommend_title(data, title_3, cat2, 1, 11, True)

    # Top 1 book recommendation title cat 1
    rec_desc_cat2_bk1 = recommend_descrip(data, title_3, cat2, 2, 12, True)
    
    rec_content_1 = pd.concat([rec_title_cat1_bk1, rec_desc_cat1_bk1
                            , rec_title_cat1_bk2, rec_desc_cat1_bk2
                            , rec_title_cat2_bk1, rec_desc_cat2_bk1])
    #rec_content_1 = pd.concat([rec_title, rec_desc_with_author])
    dic_book = rec_content_1.to_dict('records')
    return dic_book


def content_base_4(df, cat1, cat2, cat3):
    # df ==> dataframe to use
    # cat1 ==> selection category


    #### dataframe to use
    data = df

    #### dataframe to use for cat1
    data_cat1 = data.loc[df['new_tag'] == cat1].sort_values(by = 'average_rating', ascending = False)
    data_cat1.reset_index(level = 0, inplace = True)

    #### dataframe to use for cat2
    data_cat2 = data.loc[df['new_tag'] == cat2].sort_values(by = 'average_rating', ascending = False)
    data_cat2.reset_index(level = 0, inplace = True)

    #### dataframe to use for cat3
    data_cat3 = data.loc[df['new_tag'] == cat3].sort_values(by = 'average_rating', ascending = False)
    data_cat3.reset_index(level = 0, inplace = True)

    # name of title cat 1 first book
    title_1 = data_cat1['title'][0]
    print(title_1)

    # name of title cat 1 second book
    title_2 = data_cat2['title'][0]
    print(title_2)

    # name of title cat 2 
    title_3 = data_cat3['title'][0]
    print(title_3)

    # Top 1 book recommendation title cat 1
    rec_title_cat1_bk1 = recommend_title(data, title_1, cat1, 1, 13, True)

    # Top 1 book recommendation title cat 1
    rec_desc_cat1_bk1 = recommend_descrip(data, title_1, cat1, 2, 14, True)

    # Top 1 book recommendation title cat 2
    rec_title_cat2_bk2 = recommend_title(data, title_2, cat2, 1, 15, True)

    # Top 1 book recommendation title cat 2
    rec_desc_cat2_bk2 = recommend_descrip(data, title_2, cat2, 2, 16, True)
    
    # Top 1 book recommendation title cat 3
    rec_title_cat3_bk3 = recommend_title(data, title_3, cat3, 1, 17, True)

    # Top 1 book recommendation title cat 3
    rec_desc_cat3_bk3 = recommend_descrip(data, title_3, cat3, 2, 18, True)
    
    rec_content_1 = pd.concat([rec_title_cat1_bk1, rec_desc_cat1_bk1
                            , rec_title_cat2_bk2, rec_desc_cat2_bk2
                            , rec_title_cat3_bk3, rec_desc_cat3_bk3])
    #rec_content_1 = pd.concat([rec_title, rec_desc_with_author])
    dic_book = rec_content_1.to_dict('records')
    return dic_book