# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df_ratings = pd.read_csv('ratings.csv')
df_movies = pd.read_csv('movies.csv')

ratings_pt = df_ratings.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(0)

df_copy = df_ratings.copy()
df_copy['rating'] = df_copy['rating'].apply(lambda x: 0 if x > 0 else 1)

df_copy = df_copy.pivot(index = 'userId', columns = 'movieId', values = 'rating').fillna(1)

from sklearn.metrics.pairwise import cosine_similarity

# User Similarity Matrix using Cosine similarity as a similarity measure between Users
user_similarity = cosine_similarity(ratings_pt)
user_similarity[np.isnan(user_similarity)] = 0

user_pred_ratings = np.dot(user_similarity, ratings_pt)

user_final_ratings = np.multiply(user_pred_ratings, df_copy)


def userBased_Recommender(user_id):
    movies_to_recommend = 10
    if user_id in user_final_ratings.index:
        recommendations = user_final_ratings.iloc[user_id - 1].sort_values(ascending = False)[0 : movies_to_recommend]
        recommend_frame = []
        for idx in recommendations.index:
            recommend_frame.append({'Title' : df_movies.iloc[idx, 1], 'Values' : recommendations[idx]})
        df = pd.DataFrame(recommend_frame, index = range(1, movies_to_recommend + 1))
        return df
            
    else:
        return "No such user found!"