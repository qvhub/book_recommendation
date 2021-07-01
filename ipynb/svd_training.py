# Import libraries
import numpy as np
import pandas as pd

ratings = pd.read_csv('/home/quentin/Documents/simplon/20210406_moteur_de_recommandation_de_livre/source/goodbooks-10k-master/ratings.csv', index_col=False)


# %%
n_users = ratings.user_id.unique().shape[0]
n_book = ratings.book_id.unique().shape[0]
print ('Number of users = ' + str(n_users) + ' | Number of books = ' + str(n_book))

print(n_users, n_book)


id_users = ratings.user_id.unique()
id_book = ratings.book_id.unique()
print ('id of users = ' + str(n_users) + ' | id of books = ' + str(n_book))


#Ratings = ratings.pivot(index = 'user_id', columns ='book_id', values = 'rating').fillna(0)
#Ratings.head()


import joblib
import os

from scipy.sparse import csr_matrix
import scipy.sparse as sparse

Ratings = sparse.load_npz('/home/quentin/Documents/simplon/20210406_moteur_de_recommandation_de_livre/training/sparse_matrix.npz')


R = Ratings #Ratings.values
user_ratings_mean = np.mean(R, axis = 1)
Ratings_demeaned = R - user_ratings_mean.reshape(-1, 1)


Ratings_demeaned


sparsity = round(1.0 - len(ratings) / float(R.shape[0] * R.shape[1]), 3)
print ('The sparsity level of book rating dataset is ' +  str(sparsity * 100) + '%')


from scipy.sparse.linalg import svds
from scipy.linalg import sqrtm

U, sigma, Vt = svds(Ratings_demeaned, k = 30)

sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)# + user_ratings_mean.reshape(-1, 1)



print(all_user_predicted_ratings)
'''
sigma = np.diag(sigma)


s_root = sqrtm(sigma)


Usk = np.dot(U, s_root)
print(Usk)
skV = np.dot(s_root, Vt)
print(skV)


all_user_predicted_ratings = np.dot(Usk, skV) + user_ratings_mean.reshape(-1, 1)

print(all_user_predicted_ratings)
'''

