import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout
from keras.layers import LSTM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from keras.callbacks import EarlyStopping
from sklearn.metrics import pairwise_distances_argmin_min
from geopy.geocoders import Nominatim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Chargement des données des restaurants
restaurants = pd.read_csv('C:\\Users\\YOUNES\\Desktop\\Memoire\\python\\dataset\\dt\\restaurantsF.csv', low_memory=False)
# print(restaurants_df.shape)
# # Step 2: Nettoyage et préparation des données des restaurants
# restaurants_df = restaurants_df.rename(columns={'restaurant_name': 'name'})
# restaurants = restaurants_df.dropna(subset=['city', 'restaurant_link', 'name', 'original_location', 'country'])
# print(restaurants.shape)
# restaurants = restaurants.drop_duplicates(subset='name', keep=False)

# restaurants = restaurants[['restaurant_link', 'name', 'original_location',
#                             'country', 'region', 'city', 'address', 'latitude', 'longitude', 
#                             'features', 'avg_rating']]
# restaurants=restaurants.dropna(subset=['latitude', 'longitude'])
# print(restaurants.shape)
# # Step 3: Appliquer K-means aux coordonnées des restaurants
# ########## le praitraitement de model K-Means
coordinate = restaurants[['latitude', 'longitude']]
# scaler = StandardScaler()

# scaler.fit(coordinate)

# scaled_data = scaler.transform(coordinate)

# def find_best_clusters(df, maximum_K):
    
#     clusters_centers = []
#     k_values = []
    
#     for k in range(1, maximum_K):
        
#         kmeans_model = KMeans(n_clusters = k)
#         kmeans_model.fit(df)
        
#         clusters_centers.append(kmeans_model.inertia_)
#         k_values.append(k)
        
    
#     return clusters_centers, k_values


# def generate_elbow_plot(clusters_centers, k_values):
    
#     figure = plt.subplots(figsize = (12, 6))
#     plt.plot(k_values, clusters_centers, 'o-', color = 'orange')
#     plt.xlabel("Number of Clusters (K)")
#     plt.ylabel("Cluster Inertia")
#     plt.title("Elbow Plot of KMeans")
#     plt.show()

# clusters_centers, k_values = find_best_clusters(scaled_data, 12)

# generate_elbow_plot(clusters_centers, k_values)

##############################

# # Define the range of cluster numbers to evaluate
# min_clusters = 2
# max_clusters = 10

# # Create an empty list to store the silhouette scores
# silhouette_scores = []

# # Perform clustering and calculate silhouette score for each number of clusters
# for n_clusters in range(min_clusters, max_clusters+1):
#     # Initialize the K-means clustering algorithm with the current number of clusters
#     kmeans = KMeans(n_clusters=n_clusters)
#     # Fit the algorithm to the data
#     kmeans.fit(coordinate)
#     # Predict the cluster labels for each sample
#     cluster_labels = kmeans.predict(coordinate)
#     # Calculate the silhouette score
#     score = silhouette_score(coordinate, cluster_labels)
#     # Append the score to the list
#     silhouette_scores.append(score)

# # Convert the silhouette scores to a numpy array
# silhouette_scores = np.array(silhouette_scores)

# # Find the optimal number of clusters (the one with the highest silhouette score)
# optimal_n_clusters = np.argmax(silhouette_scores) + min_clusters

# # Plot the silhouette scores
# plt.plot(range(min_clusters, max_clusters+1), silhouette_scores)
# plt.xlabel("Number of Clusters")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Score vs. Number of Clusters")
# plt.show()

# # Print the optimal number of clusters
# print("Optimal number of clusters:", optimal_n_clusters)

# for r in restaurants['avg_rating']:
#     if r == None:
#         restaurants.at[restaurants.index[r], "avg_rating"] = '3'

# restaurants = restaurants.drop_duplicates(subset='name', keep=False)
# restaurants['id'] = range(len(restaurants))
# restaurants.to_csv("dataset\\dt\\restaurantsF.csv", index=False)



kmeans =KMeans(init="k-means++",n_clusters = 4 , n_init=10 , max_iter=300) #le meilleur nbr de cluster cest 6 dapres le graphe

kmeans.fit(coordinate)

centroids = kmeans.cluster_centers_

restaurants['cluster_label'] = kmeans.labels_
# # print('////////')

def get_coordinates(city):
    geolocator = Nominatim(user_agent="recomendation_app")
    location = geolocator.geocode(city)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        return latitude, longitude
    else:
        return None

def requirementbased(city,features):
    restaurants['city']=restaurants['city'].str.lower()
    restaurants['features']=str(restaurants['features']).lower()
    # features=features.lower()
    features_tokens=word_tokenize(features)  
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f1_set = {w for w in features_tokens if not w in sw}
    f_set=set()

    coordinates=get_coordinates(city)
    lat=coordinates[0]
    long=coordinates[1]

    new_data = np.array([[lat, long]])
    
    cluster=kmeans.predict(new_data)

    h=restaurants[restaurants['cluster_label'] == cluster[0]]

    for se in f1_set:
        f_set.add(lemm.lemmatize(se))
    reqbased=h[h['city']==city.lower()]
    
    reqbased=reqbased.set_index(np.arange(reqbased.shape[0]))
    l1 =[];l2 =[];cos=[];
    #print(reqbased['roomamenities'])
    for i in range(reqbased.shape[0]):
        temp_tokens=word_tokenize(str(reqbased['features'][i]))
        temp1_set={w for w in temp_tokens if not w in sw}
        temp_set=set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        #print(rvector)
        cos.append(len(rvector))
    reqbased['similarity']=cos
    reqbased=reqbased.sort_values(by='similarity',ascending=False)
    reqbased.drop_duplicates(subset='name',keep='first',inplace=True)
    m=reqbased[['id','name','city', 'address','features', 'avg_rating','similarity','image']]
    return m.head(10)

# print(requirementbased('paris','wifi'))
# print(requirementbased('paris','wifi tv air conditioner'))
# print(restaurants.columns)

from math import sin, cos, sqrt, atan2, radians
R=6373.0#Earth's Radius
sw = stopwords.words('english')
lemm = WordNetLemmatizer()

def hybrid(city,features):
    restaurants['city']=restaurants['city'].str.lower()
    restaurants['features']=str(restaurants['features']).lower()
    features=features.lower()
    features_tokens=word_tokenize(features)
    f1_set = {w for w in features_tokens if not w in sw}
    f_set=set()
    for se in f1_set:
        f_set.add(lemm.lemmatize(se))
    
    coordinates=get_coordinates(city)
    lat1=48.868424
    long1=2.345659

    new_data = np.array([[lat1, long1]])
    
    cluster=kmeans.predict(new_data)

    h=restaurants[restaurants['cluster_label'] == cluster[0]]

    dist=[]
    
    lat1=radians(float(lat1))
    long1=radians(float(long1))


    hybridbase=h
    # hybridbase['city']=hybridbase['city'].str.lower()
    # hybridbase.loc[:, 'city'] = hybridbase['city'].str.lower()
    hybridbase=hybridbase.set_index(np.arange(hybridbase.shape[0]))

    for i in range(hybridbase.shape[0]):
        lat2=radians(hybridbase['latitude'][i])
        long2=radians(hybridbase['longitude'][i])
        dlon = long2 - long1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        dist.append(distance)
    hybridbase['distance']=dist
    hybridbase=hybridbase.sort_values(by='distance',ascending=True)
    
    hybridbase=hybridbase.head(15)
    
    hybridbase=hybridbase.set_index(np.arange(hybridbase.shape[0]))
    
    coss=[]
    for i in range(hybridbase.shape[0]):
        temp_tokens=word_tokenize(hybridbase['features'][i])
        temp1_set={w for w in temp_tokens if not w in sw}
        temp_set=set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        coss.append(len(rvector))
    
    hybridbase['similarity']=coss
    x=hybridbase.sort_values(by='similarity',ascending=False).head(10)

    m=x[['id','name','city', 'address','features', 'avg_rating','similarity','image','distance']]
    return m


print(hybrid('paris','wifi'))



def getrestaurant(id):
    x = restaurants[restaurants['id'] == id]
    return x



######## la visualisation de dataset
# dataset=restaurants
# # Extraction des coordonnées de latitude et de longitude
# latitude = dataset['latitude']
# longitude = dataset['longitude']
# cluster_labels = dataset['cluster_label']

# # Extraction des coordonnées des centroids de chaque cluster
# centroids = dataset.groupby('cluster_label').mean().reset_index()  # Réinitialisation de l'index
# centroid_latitude = centroids['latitude']
# centroid_longitude = centroids['longitude']

# # Affichage des points par cluster avec les centroids
# plt.figure(figsize=(10, 6))
# plt.scatter(longitude, latitude, c=cluster_labels, cmap='viridis', alpha=0.5)
# plt.scatter(centroid_longitude, centroid_latitude, marker='X', c='red', s=100, label='Centroids')
# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.title('Carte de la dataset restaurant par cluster')
# plt.legend()
# plt.grid(True)
# plt.show()


# dataset=restaurants
# # Extraction des coordonnées de latitude et de longitude
# latitude = dataset['latitude']
# longitude = dataset['longitude']

# # Affichage des points sur la carte
# plt.figure(figsize=(10, 6))
# plt.scatter(longitude, latitude, c='b', alpha=0.5)
# plt.xlabel('longitude')
# plt.ylabel('latitude')
# plt.title('Carte de la dataset restaurant')
# plt.grid(True)
# plt.show()
