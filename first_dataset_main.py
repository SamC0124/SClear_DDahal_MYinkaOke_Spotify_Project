# Final Project Main Function: Finding Greatest Predictors for Dancability and Valence of the Most Popular Songs on Spotify
# Creators: Michael Yinka'Oke, Sam Clear, Diwas Dahal
# Start Date: March 21st, 2024

# This file currently acts as the main dataset inspecting file of the program, but will in the future just be
# used for running the main bulk of objects in the program.

# Essential Packages for inspecting the data
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import string
import sklearn as sk
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn import tree

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


# Function for collecting the false positives and the true positives from a classifiers confusion matrix.
# Input: Two lists for actual and predicted values.
# Output: Two totals for number of True positives and False positives.
def retrieve_positive_classifications(p_y_test, p_y_pred):
    conf_matrix = confusion_matrix(p_y_test, p_y_pred)
    true_positive = conf_matrix[0][0]
    false_positive = conf_matrix[0][1]
    return true_positive, false_positive


if __name__ == '__main__':
    ## Data Cleaning
    music_data = pd.read_csv('data/song_data.csv', index_col=0).dropna()

    # Try removing all popularity values that are equal to zero. These songs may have never been evaluated, which may
    # misrepresent the true metrics of an unpopular song.
    music_data = music_data[music_data['popularity'] > 0]

    # Between 1-3 different entries don't have an artist, album name, or track_name. Are these columns what we want to
    # use to evaluate the relationship between different characteristics in songs, Popularity, and Dancability?
    # Another thing to note is that some tracks are entered into the dataset more than once. These songs will be
    # overrepresented in the dataset, so we can just take unique values by track ID.
    print(music_data.info())
    print(music_data.describe())
    print("Some of the data is unnamed, has no popularity, tempo, or time signature. This data is likely null.")

    cont_pop_data = music_data.drop_duplicates(subset=['track_name'])
    cont_pop_data.loc[:, 'duration_s'] = cont_pop_data.loc[:, 'duration_ms'] / 1000
    cont_pop_data = cont_pop_data.drop(columns=['track_id', 'key']).dropna()

    # Load data and drop unnecessary columns and rows with null values
    unique_data = music_data.drop(columns=['track_id']).dropna()

    # Binarizing popularity in the dataset (Song is popular or '1' if popularity > 75, else song is labeled '0')
    cleared_data = unique_data
    cleared_data['popularity'] = cleared_data['popularity'].apply(lambda x: 1 if x > 75 else 0)

    # Convert True to 1 and False to 0 in the "popularity" column
    cleared_data['explicit'] = cleared_data['explicit'].astype(int)
    # Remove duplicates based on track name
    cleared_data = cleared_data.drop_duplicates(subset=['track_name'])
    # Convert milliseconds to seconds
    cleared_data = cleared_data.rename(columns={'duration_ms': 'duration_s'})
    cleared_data.loc[:, 'duration_s'] = cleared_data.loc[:, 'duration_s'] / 1000

    print(f"Data Classified as \"Popular\": {len(cleared_data[cleared_data['popularity'] == 1])}\nData Classified as \"Non-Popular\" {len(cleared_data[cleared_data['popularity'] == 0])}")
    print("There is an extreme class imbalance for the positive class against the negative class. This should be addressed with an undersampling of the negative class data.")

    # Undersampling the negative class for the data
    undersampled_negative_class = cleared_data[cleared_data['popularity'] == 0].to_numpy().tolist()
    undersampled_negative_class = random.sample(population=undersampled_negative_class, k=len(cleared_data[cleared_data['popularity'] == 1]))
    print(len(undersampled_negative_class), len(cleared_data[cleared_data['popularity'] == 1]))
    print("Data should now be balanced, however a large amount of data was lost. We may decrease the minimum popularity score needed to be labeled \'popular\'")

    # Placing back into a pandas dataframe
    undersampled_data = np.concatenate([undersampled_negative_class, cleared_data[cleared_data['popularity'] == 1].to_numpy()])
    undersampled_dataframe = pd.DataFrame(undersampled_data, columns=cleared_data.columns)

    # List of columns you want to keep (Including Key)
    columns_to_keep = ['popularity', 'duration_s', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    columns_to_get_mean = ['duration_s', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                           'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']

    ## Prior Graphing, Gathering Insights
    # Computing correlations of data with highest popularity value
    '''we consider a song is popular if it's popularity determined by the number of time it was played is greater than 75
    * also if we want we can increase the number of popularity for better accuracy'''
    pop_data = music_data[music_data['popularity'] > 75]
    pop_columns = ['artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'danceability', 'energy',
                'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                'time_signature', 'track_genre']
    plt.imshow(pop_data.corr(), cmap="PuBu")
    plt.xticks(ticks=range(len(pop_columns)), labels=pop_columns, rotation=90)
    plt.yticks(ticks=range(len(pop_columns)), labels=pop_columns)
    plt.title("Correlations between Characteristics of Songs with High Popularities")
    plt.tight_layout()
    plt.colorbar()
    plt.close()

    ## SVM with Soft Margin (Allow for missclassification at a low cost, essential for our imperfect dataset)
    ## With the current hour of this work, this program has been assisted by ChatGPT, plotting will be done on our own.
    # Collect portion of dataset for X (features to gauge popularity by) and Y (popularity classification
    X = undersampled_dataframe[columns_to_get_mean]
    y = undersampled_dataframe['popularity']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.shape, y_train.shape)

    # Collect portion of dataset for X (features to gauge popularity by) and Y (popularity classification
    X_old = cleared_data[columns_to_get_mean]
    y_old = cleared_data['popularity']

    # Split data into train and test sets
    X_train_old, X_test_old, y_train_old, y_test_old = train_test_split(X_old, y_old, test_size=0.2, random_state=0)
    print(X_train_old.shape, y_train_old.shape)

    # Create SVM classifier with soft-margin (C=1)
    # Measure differences in SVM Accuracy by kernel
    print("Creating Classifiers...")
    svm_classifier_linear = SVC(kernel='linear', degree=3, C=1, random_state=1)
    svm_classifier_sgd = SGDClassifier(max_iter=250, random_state=1)
    svm_classifier_poly = SVC(kernel='poly', degree=3, C=1, random_state=1)
    svm_classifier_sigmoid = SVC(kernel='sigmoid', degree=3, C=1, random_state=1)
    svm_classifier_rbf = SVC(degree=3, C=1, random_state=1)

    # Train the classifiers
    print("Training Classifiers...")
    svm_classifier_linear.fit(X_train, y_train)
    svm_classifier_sgd.fit(X_train, y_train)
    svm_classifier_poly.fit(X_train, y_train)
    svm_classifier_sigmoid.fit(X_train, y_train)
    svm_classifier_rbf.fit(X_train, y_train)

    # Make predictions on test data
    print("Classifying unseen data based on trained classifiers...")
    y_pred_linear = svm_classifier_linear.predict(X_test)
    y_pred_sdg = svm_classifier_sgd.predict(X_test)
    y_pred_poly = svm_classifier_poly.predict(X_test)
    y_pred_sigmoid = svm_classifier_sigmoid.predict(X_test)
    y_pred_rbf = svm_classifier_rbf.predict(X_test)

    # Calculate accuracy of each model
    print("Processing the accuracy of each classifier, compared to actual data values...")
    accuracy_linear = accuracy_score(y_test, y_pred_linear)
    accuracy_sgd = accuracy_score(y_test, y_pred_sdg)
    accuracy_poly = accuracy_score(y_test, y_pred_poly)
    accuracy_sigmoid = accuracy_score(y_test, y_pred_sigmoid)
    accuracy_rbf = accuracy_score(y_test, y_pred_rbf)

    print("Model Accuracy by Kernel/Method")
    print(f"Linear Kernel: {accuracy_linear}")
    print(f"Linear SGD Kernel: {accuracy_sgd}")
    print(f"Polynomial Kernel: {accuracy_poly}")
    print(f"Sigmoid Kernel: {accuracy_sigmoid}")
    print(f"RBF Kernel: {accuracy_rbf}")

    # What classification errors were made in each type of model? Collect true positives against false positives for each type of kernel
    svc_linear_true_positive, svc_linear_false_positive = retrieve_positive_classifications(y_test, y_pred_linear)
    svc_sgd_true_positive, svc_sgd_false_positive = retrieve_positive_classifications(y_test, y_pred_linear)
    svc_poly_true_positive, svc_poly_false_positive = retrieve_positive_classifications(y_test, y_pred_linear)
    svc_sigmoid_true_positive, svc_sigmoid_false_positive = retrieve_positive_classifications(y_test, y_pred_linear)
    svc_rbf_true_positive, svc_rbf_false_positive = retrieve_positive_classifications(y_test, y_pred_linear)

    positive_class_data = pd.DataFrame({"Kernel Type": ["Linear", "Linear SGD", "Polynomial", "Sigmoid", "RBF"],
                                        "True Positive": [svc_linear_true_positive, svc_sgd_true_positive, svc_poly_true_positive, svc_sigmoid_true_positive, svc_rbf_true_positive],
                                        "False Positive": [svc_linear_false_positive, svc_sgd_false_positive, svc_poly_false_positive, svc_sigmoid_false_positive, svc_rbf_false_positive]})
    print(positive_class_data[["True Positive", "False Positive"]])
    print(positive_class_data[["True Positive", "False Positive"]].to_numpy())

    # Comparing TP/FP on Two-layer Bar Plot
    plt.hist(x=np.array(positive_class_data[["True Positive", "False Positive"]].to_numpy()), histtype='bar', density=True, color=['green', 'red'], label=["True Positive", "False Positive"])
    plt.xlabel("Kernel Type for SVC, divided by TP and FP Totals")
    plt.ylabel("Totals Correct/Incorrect Classifications")
    plt.title("Comparing Kernel Classifications by True/False Positive Totals")
    plt.legend()
    plt.show()
    exit()

    # What are the equations computing popularity from the two highest support vectors (values of each feature)?
    most_significant_vector = svm_classifier_poly.support_vectors[0]
    second_most_significant_vector = svm_classifier_poly.support_vectors[1]

    # Currently the accuracy for the model is extremely high, but we don't know why.
    # What can we do to evaluate whether the model has not overfit the data or not?
    # How can we view how effectively the model has classified the data?
    print(svm_classifier_poly.n_features_in_)
    print(svm_classifier_poly.support_vectors_)
    plt.imshow(svm_classifier_poly.support_vectors_)
    plt.title("Support vector ")
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Calculate average of specified columns
    '''using that data we find the average the durations_ms, danceability, 
    energy, loudness and any other factors that can help us make a model to predict the popularity of a song'''

    average_data = pop_data[
        ['duration_ms', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness',
         'liveness',
         'valence', 'tempo', 'time_signature']].mean()

    '''using the average we set the upper limit/threshold to predict the popularity of a song
    * once our model is created we can see what features make the song popular 
    * we would most likely need create some sort of classification to check the features '''

    plt.scatter(data=pop_data, x='tempo', y='popularity')
    plt.show()
    print(average_data)

    ## Decision Tree Modeling
    # First Decision Tree Model - All features
    # This Decision Tree Classifier Program was adapted from the following site: https://scikit-learn.org/stable/modules/tree.html
    X = cleared_data[columns_to_get_mean]
    Y = cleared_data['popularity']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    print(tree.plot_tree(clf))

    import graphviz

    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data)
    graph.render("spotify_pop_data1_no_lim")
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=columns_to_get_mean,
                                    class_names=['popular', 'not_popular'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = graphviz.Source(dot_data)

    ## K-Means Clustering: Not Fit for Model
    # pop_clusters = KMeans(n_clusters=5, max_iter=500, random_state=42).fit(cont_pop_data[['loudness', 'danceability']])
    # cont_pop_data["popularity_groups"] = pop_clusters.labels_
    #
    # plt.scatter(data=unique_data, x="popularity", y="danceability", c="popularity_groups")
    # plt.show()
    #
    # first_group = cont_pop_data[cont_pop_data["popularity_groups"] == 0]
    # second_group = cont_pop_data[cont_pop_data["popularity_groups"] == 1]
    # third_group = cont_pop_data[cont_pop_data["popularity_groups"] == 2]
    # fourth_group = cont_pop_data[cont_pop_data["popularity_groups"] == 3]
    # fifth_group = cont_pop_data[cont_pop_data["popularity_groups"] == 4]
    #
    # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5)
    # ax1.hist(x=first_group['popularity'], bins=8, density=True, histtype='bar')
    # ax2.hist(x=second_group['popularity'], bins=8, density=True, histtype='bar')
    # ax3.hist(x=third_group['popularity'], bins=8, density=True, histtype='bar')
    # ax4.hist(x=fourth_group['popularity'], bins=8, density=True, histtype='bar')
    # ax5.hist(x=fifth_group['popularity'], bins=8, density=True, histtype='bar')
    # fig.tight_layout()
    # plt.show()

    ## K-Nearest-Neighbors: Fit adequately for model
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
    # Create a box plot
    sns.set(style="whitegrid")
    g = sns.FacetGrid(long_data, col='feature', col_wrap=2, margin_titles=True,
                      xlim=(long_data['value'].min(), long_data['value'].max()))
    g.map(sns.boxplot, 'value', 'popularity', 'popularity', order=['popular', 'not_popular'],
          hue_order=['popular', 'not_popular'], palette={"popular": "tomato", "not_popular": "cyan"})
    plt.title("")
    plt.xlabel("")
    plt.ylabel("")

    # Remove the legend
    plt.legend().remove()
    plt.xlim(left=0, right=1)
    plt.show()

    # Select numeric columns
    numeric_cols = cleared_data.select_dtypes(include='number')
    # Min-max normalization
    pop_norm = (numeric_cols - numeric_cols.min()) / (numeric_cols.max() - numeric_cols.min())
    # Standardization
    pop_stan = (numeric_cols - numeric_cols.mean()) / numeric_cols.std()
    # Create knn_results DataFrame
    knn_results = pd.DataFrame({'k': range(1, 6), 'pop_norm': [-1] * 5, 'pop_stan': [-1] * 5})
    # Displaying the knn_results DataFrame
    print(knn_results)
    # Convert 'pop_norm' and 'pop_stan' columns to float64
    knn_results['pop_norm'] = knn_results['pop_norm'].astype(float)
    knn_results['pop_stan'] = knn_results['pop_stan'].astype(float)
    # Fit KNN Algorithm for normalized data
    for i in range(len(knn_results)):
        knn = KNeighborsClassifier(n_neighbors=knn_results.loc[i, 'k'])
        loop_knn = cross_val_predict(knn, pop_norm, cleared_data['popularity'], cv=5)
        loop_norm_cm = confusion_matrix(loop_knn, cleared_data['popularity'])
        accuracy = round(accuracy_score(loop_knn, cleared_data['popularity']), 2)
        print(f"Accuracy for k={knn_results.loc[i, 'k']} with normalized data: {accuracy}")

        # Debugging print
        knn_results.loc[i, 'pop_norm'] = accuracy
        # Fit KNN Algorithm for standardized data
        knn = KNeighborsClassifier(n_neighbors=knn_results.loc[i, 'k'])
        loop_knn2 = cross_val_predict(knn, pop_stan, cleared_data['popularity'], cv=5)
        accuracy2 = round(accuracy_score(loop_knn2, cleared_data['popularity']), 2)
        print(f"Accuracy for k={knn_results.loc[i, 'k']} with standardized data: {accuracy2}")

        # Debugging print
        knn_results.loc[i, 'pop_stan'] = accuracy2
        # Displaying the first 10 rows of knn_results DataFrame
        # print(knn_results.head(10))
        # long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')

        # # # Create the plot
        # plt.figure(figsize=(10, 6))
        # sns.lineplot(data=long_knn_results, x='k', y='accuracy', hue='rescale_method')

        # # # Set labels and title
        # plt.xlabel('Choice of K') # plt.ylabel('Accuracy')
        # plt.title('KNN Algorithm Performance')
        # plt.legend(title='Rescale Method', labels=['Normalized', 'Standardized'])

        # # # Set scale for x-axis and y-axis # plt.xticks(range(1, 6))
        # plt.yticks(np.arange(0.95, 1.0, 0.01), labels=[f'{i:.2f}%' for i in np.arange(0.95, 1.0, 0.01)])

        # # # Adjust legend position
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # # # Show plot
        # plt.grid(True)
        # plt.show()
        long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')
        # Select the rows with the maximum accuracy for each rescale method
        max_accuracy_rows = long_knn_results.loc[long_knn_results.groupby('rescale_method')['accuracy'].idxmax()]

        # Displaying the knn_results DataFrame
        print(knn_results)

    long_knn_results = knn_results.melt(id_vars='k', var_name='rescale_method', value_name='accuracy')

    # Select the rows with the maximum accuracy for each rescale method
    max_accuracy_rows = long_knn_results.loc[long_knn_results.groupby('rescale_method')['accuracy'].idxmax()]
    print(max_accuracy_rows)
    # Ensure 'popularity' is included in pop_norm
    pop_norm['popularity'] = cleared_data['popularity']
    # Splitting data into features (X) and target (y)
    X = pop_norm.drop(columns=['popularity'])
    # Exclude 'popularity' from features
    y = pop_norm['popularity']
    # Use 'popularity' as the target column
    # Splitting data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Initializing KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=2)
    # Fitting the classifier on the training data
    knn_classifier.fit(X_train, y_train)
    # Predicting on the test data
    y_pred = knn_classifier.predict(X_test)
    # Calculating confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    # Displaying confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn_classifier.classes_)
    disp.plot()