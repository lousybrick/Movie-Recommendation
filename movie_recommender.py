import numpy as np
import pandas as pd
import warnings 

# Filters all of warnings
warnings.filterwarnings('ignore')


# Getting the Dataset
column_names = ["user_id", "item_id", "rating", "timestamp"]


# Reading the csv file of the user data
df = pd.read_csv("ml-100k/u.data", sep = "\t", names = column_names)


# Reading the csv of the movie titles
movies_titles = pd.read_csv("ml-100k/u.item", sep = "\|", header = None)
movies_titles = movies_titles[[0,1]]
movies_titles.columns = ['item_id', 'title']


# Merging my movies_titles dataframe to df dataframe
df = pd.merge(df, movies_titles, on = "item_id")

# Making a ratings dataframe 
ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num of ratings'] = pd.DataFrame(df.groupby('title').count()['rating'])


# Creating the Movie recommendation 

# Making a movie matrix between user_id and titles with ratings as the correlated values
movie_matrix = df.pivot_table(index = "user_id", columns = "title", values = "rating")


# Final Predict Function
def predict_movies(movie_name):
    movie_user_ratings = movie_matrix[movie_name]
    similar_to_movie = movie_matrix.corrwith(movie_user_ratings)
    
    corr_movie = pd.DataFrame(similar_to_movie, columns = ['Correlation'])
    corr_movie.dropna(inplace = True)
    
    corr_movie = corr_movie.join(ratings['num of ratings'])
    predictions = corr_movie[corr_movie['num of ratings'] > 100].sort_values('Correlation', ascending = False)
    
    return predictions


#Taking user input 
movie_name = input("Enter the name of the movie in the following way - Title (Year): ")
number_of_movies = int(input("Enter the number of recommendations you want to see:"))

predictions = predict_movies(movie_name)

print(predictions.head(number_of_movies))



