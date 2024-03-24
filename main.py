# Final Project Main Function: Finding Greatest Predictors for Dancability and Valence of the Most Popular Songs on Spotify
# Creators: Michael Yinka'Oke, Sam Clear, Diwas Dahal
# Start Date: March 21st, 2024

# This file currently acts as the main dataset inspecting file of the program, but will in the future just be
# used for running the main bulk of objects in the program.

# Essential Packages for inspecting the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import os
import time

# Main function
# Questions we want to answer: What features are the most effective at predicting whether a song is popular on Spotify?
# - We have seen that genres of songs have changed in popularity over time, but we may be able to pair genre with other
#   musical aspects to find a better predictor for popularity.
# - What features are the best at predicting popularity?
# This can be tested with giving machine learning models datasets of different features, then see which subset of features is the best for predicting this trait.
# However, we also have a prompt offered around predicting what songs people would be interested based on their danceability and valence scores:
# - What do we need to know what songs would fit a person's danceability and valence scores? What does this have to do with popularity scores?
# - A person isn't necessarily going to like a song based on what everyone else likes in a song, people's perception's
#   of a songs danceability can be variable for one particular song. This can't really be shown with our project, since
#   we have the average popularity scores for each popular song, but I believe it would confound our project's ability
#   to return songs that a person would actually like.


if __name__ == '__main__':
    music_data = pd.read_csv('data/song_data.csv', index_col=0).dropna()
    # print(data.describe())
    print("Some of the data is unnamed, has no popularity, tempo, or time signature. This data is likely null.")

    # Between 1-3 different entries don't have an artist, album name, or track_name. Are these columns what we want to
    # use to evaluate the relationship between different characteristics in songs, Popularity, and Dancability?
    # Another thing to note is that some tracks are entered into the dataset more than once. These songs will be
    # overrepresented in the dataset, so we can just take unique values by track ID.
    # print(data.info())

    # Load data and drop unnecessary columns and rows with null values
    unique_data = music_data.drop(columns=['track_id', 'key']).dropna()

    # Code for Timer to Start and Stop Showing Plots: https://stackoverflow.com/questions/30364770/how-to-set-timeout-to-pyplot-show-in-matplotlib
    # def close_plot():
    #     plt.close()
    #     return
    
    # Initializing timer
    # fig = plt.figure()
    # timer = fig.canvas.new_timer(interval=1000) # Figure waits 5000 millisecond before calling a callback event
    # timer.add_callback(close_plot)

    # # Correlation Matrix for the data
    # plt.imshow(unique_data.corr(), cmap="PuBu")

    # # Show plot for 5 seconds
    # # timer.start()
    # plt.show()

    # Looking for relationship from clusters based on dancability, explicity, and instrumentalness.
    clusters = KMeans(n_clusters=5, max_iter=1000, random_state=1).fit(unique_data[['danceability', 'explicit', 'instrumentalness']])
    print(clusters.labels_)
    unique_data["possible_groups"] = clusters.labels_
    print(unique_data.corr())

    # If we make 4 clusters based on popularity, what characteristics are shared between songs in each group?
    pop_clusters = KMeans(n_clusters=5, max_iter=500, random_state=42).fit(unique_data[['popularity', 'danceability']])
    unique_data["popularity_groups"] = pop_clusters.labels_
    print(len(unique_data['popularity']), len(unique_data['danceability']))

    plt.scatter(data=unique_data, x="popularity", y="danceability", c="popularity_groups")
    plt.show()

    first_group = unique_data[unique_data["popularity_groups"] == 0]
    second_group = unique_data[unique_data["popularity_groups"] == 0]
    third_group = unique_data[unique_data["popularity_groups"] == 0]
    fourth_group = unique_data[unique_data["popularity_groups"] == 0]

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 1, ncols=4)
    ax1.hist(x=first_group['danceability'], bins=8, density=True, histtype='bar')
    ax2.hist(x=second_group['danceability'], bins=8, density=True, histtype='bar')
    ax3.hist(x=third_group['danceability'], bins=8, density=True, histtype='bar')
    ax4.hist(x=fourth_group['danceability'], bins=8, density=True, histtype='bar')
    fig.tight_layout()
    plt.show()

    # Computing correlations of data with highest popularity value
    '''we consider a song is popular if it's popularity determined by the number of time it was played is greater than 75
    * also if we want we can increase the number of popularity for better accuracy'''
    pop_data = unique_data[unique_data['popularity'] > 80]
    plt.imshow(pop_data.corr(), cmap="PuBu")
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], ['artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'], rotation=45)
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], ['artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'])
    plt.title("Correlations between Characteristics of Songs with High Popularities")
    plt.tight_layout()
    plt.show()

    # Calculate average of specified columns
    '''using that data we find the average the durations_ms, danceability, 
    energy, loudness and any other factors that can help us make a model to predict the popularity of a song'''

    average_data = pop_data[['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']].mean()

    '''using the average we set the upper limit/threshold to predict the popularity of a song
    * once our model is created we can see what features make the song popular 
    * we would most likely need create some sort of classification to check the features '''

    plt.scatter(data=pop_data, x='tempo', y='popularity')
    plt.show()
    print(average_data)

    # Additional Code from Diwas
    '''using the average we can make model like linear regression, decision tree, or forest''' 
    cleared_data = pd.read_csv('data/song_data.csv').drop(columns=['track_id', 'key']).dropna(how='any') 
    cleared_data['popularity'] = cleared_data['popularity'].apply(lambda x: 'popular' if x > 75 else 'not_popular') 
    
    # Convert True to 1 and False to 0 in the "popularity" column 
    cleared_data['explicit'] = cleared_data['explicit'].astype(int) 
    # Remove duplicates based on track name 
    cleared_data = cleared_data.drop_duplicates(subset=['track_name']) 
    # Convert milliseconds to seconds 
    cleared_data['duration_s'] = cleared_data['duration_ms'] / 1000 

    # List of columns you want to keep 
    columns_to_keep = ['popularity','duration_s', 'explicit', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'] 
    columns_to_get_mean = ['duration_s', 'explicit', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'] 
    # Filtering columns 
    cleared_data = cleared_data[columns_to_keep] 
    # Reshape the data to long format 
    long_data = pd.melt(cleared_data, id_vars=['popularity'], var_name='feature', value_name='value') 
    # Calculate the mean for each feature and diagnosis 
    means = long_data.groupby(['feature', 'popularity'])['value'].mean().reset_index() 
    # Reshape the data back to wide format 
    wide_means = means.pivot(index='feature', columns='popularity', values='value') 
    # Print the result 
    print(wide_means)
    #Create a box plot 
    sns.set(style="whitegrid") 
    g = sns.FacetGrid(long_data, col='feature', col_wrap=2, margin_titles=True, xlim=(long_data['value'].min(), long_data['value'].max())) 
    g.map(sns.boxplot, 'value', 'popularity', 'popularity', order=['popular', 'not_popular'], hue_order=['popular', 'not_popular'], palette={"popular": "tomato", "not_popular": "cyan"}) 
    plt.title("") 
    plt.xlabel("") 
    plt.ylabel("") 
    
    # Remove the legend 
    plt.legend().remove() 
    plt.xlim(left=0, right=1)
    plt.show()
