from typing import List

from sqlalchemy.orm import Session
from fastapi import Depends, FastAPI, HTTPException, Request, Form

from database import *
from starlette.responses import RedirectResponse

import pandas as pd
import numpy as np

import uvicorn

from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from modules.preprocessing.preprocessing import preprocessing

from modules.new_user.content_base import content_base_1, content_base_2, content_base_3, content_base_4

templates = Jinja2Templates(directory="api/templates/")

# Get ratings into DB

query_r = sqlal.select([db_ratings])

ResultProxy_r = connection.execute(query_r)

ResultSet_r = ResultProxy_r.fetchall()

df_ratings = pd.DataFrame(ResultSet_r)

# Convert sql to DataFrame

df_ratings = df_ratings.rename(columns={0: 'user_id', 1: 'book_id', 2: 'rating'})

# Get ratings into DB

query_books = sqlal.select([db_books_infos])

ResultProxy_books = connection.execute(query_books)

ResultSet_books = ResultProxy_books.fetchall()

df_books_infos = pd.DataFrame(ResultSet_books)

df_books_infos = df_books_infos.rename(columns={0: 'id',
                                                1: 'goodreads_book_id ',
                                                2: 'tag_id',
                                                3: 'count',
                                                4: 'tag_name',
                                                5: 'new_tag',
                                                6: 'description',
                                                7: 'book_id',
                                                8: 'best_book_id',
                                                9: 'work_id',
                                                10: 'books_count',
                                                11: 'isbn',
                                                12: 'isbn13',
                                                13: 'authors',
                                                14: 'original_publication_year',
                                                15: 'original_title',
                                                16: 'title',
                                                17: 'language_code ',
                                                18: 'average_rating',
                                                19: 'ratings_count',
                                                20: 'work_ratings_count',
                                                21: 'work_text_reviews_count',
                                                22: 'ratings_1',
                                                23: 'ratings_2',
                                                24: 'ratings_3',
                                                25: 'ratings_4',
                                                26: 'ratings_5',
                                                27: 'image_url',
                                                28: 'small_image_url',
                                                })


df_books_infos = preprocessing(df_books_infos)

print("Connecting...")


'''
for i in range (1, 10001):
    df_ratings[i] = df_ratings[i].replace(np.nan, 0)

print(df_ratings)
'''

app = FastAPI()


@app.on_event("startup")
async def startup():

    print("Connecting...")


@app.get('/hello')
def read_form():

    return 'hello world'

# N E W - U S E R 
@app.get("/user")
async def form_post(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request}) #

# U S E R - R E S U L T
@app.post("/result")
async def test(request: Request, num: int = Form(...)):
    #content_1 = content_result_1(num, df_selected)
    result = function_knn(df_ratings, num)

    return templates.TemplateResponse('result.html', context={'request': request, 'result': result})

# N E W - U S E R 
@app.get("/")
async def form_post(request: Request):

    book_title = df_books_infos['title'].unique()


    tag_name = df_books_infos['new_tag'].unique()

    list_tag_name = []

    for i in tag_name:
        if i == '':
            pass
        else:
            list_tag_name.append(i)
    

    #list_category = ['cat1','cat2','cat3','cat4','cat5']

    return templates.TemplateResponse('new_user.html', context={'request': request, 'cat':list_tag_name, 'book_title':book_title})

# B O O K - L I K E D
@app.post("/new-user-book-liked-result")
async def result_new_user(request: Request, name_book_liked: str = Form(...)):
    print('----------------')
    print(name_book_liked)
    print('----------------')
    result = content_base_1(df_books_infos, name_book_liked)
    print('----------------')
    print('book liked form')
    print('----------------')
    return templates.TemplateResponse('result.html', context={'request': request,'result': result})


# C A T E G O R I E S - S E L E C T I O N
@app.post("/new-user-cat-result")
async def result_new_user(request: Request, cat_form: list = Form(...)):

    print(cat_form)
    print('*** cat lsite first ***')
    print(cat_form[0])
    print('*** type ***')
    print(type(cat_form[0]))

    if len(cat_form) == 1:
        result = content_base_2(df_books_infos, cat_form[0])
    elif len(cat_form) == 2:
        result = content_base_3(df_books_infos, cat_form[0], cat_form[1])
    elif len(cat_form) == 3:
        result = content_base_4(df_books_infos, cat_form[0], cat_form[1], cat_form[2])
    else:
        result = "Maximum 3 categories"
    return templates.TemplateResponse('result.html', context={'request': request,'result': result})

if __name__ == "__main__":
 uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
