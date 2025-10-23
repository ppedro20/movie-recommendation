import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title_from_index(index, df):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title, df):
    return df[df.title == title]["index"].values[0]

# Load the dataset
df = pd.read_csv("data/movie_dataset.csv")
#print(df.columns)

#select features
features = ['keywords','cast','genres','director']

#column to combine features 
for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']
    
df["combined_features"] = df.apply(combine_features, axis=1)
#print (df["combined_features"].head())

#count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])

#compute the cosine similarity based on the count matrix
similarity_scores = cosine_similarity(count_matrix)
#print(similarity_scores)

#get the title of the movie from user
movie_user_likes = "My Big Fat Greek Wedding 2"

#find the index of this movie from its title
movie_index = get_index_from_title(movie_user_likes, df)

#get a list of similar movies in form of (movie index, similarity score)
similar_movies = list(enumerate(similarity_scores[movie_index]))

#sort the movies based on similarity scores
sorted_similar_movies = sorted(similar_movies, key=lambda x:x[1], reverse=True)

#print the titles of first 5 similar movies
sorted_similar_movies = sorted_similar_movies[1:11]

print("Top 5 similar movies to " + movie_user_likes + " are:\n")
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0], df))
    
