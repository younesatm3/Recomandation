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

# Charger les données des hôtels
hotel=pd.read_csv('C:\\Users\\YOUNES\\Desktop\\Memoire\\python\\dataset\\dt\\hotelF.csv',delimiter=',')

coordinate = hotel[['latitude', 'longitude']]

# c=hotel.drop_duplicates(subset='hotelname')
# print(c.shape)

####### Pour la performance de model on doit choisie le meilleur cluster
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


# print(hotel.shape)
# hotel= hotel.drop_duplicates(subset='hotelname')
# print(hotel.shape)
# restaurants = hotel.drop_duplicates(subset='name', keep=False)
# hotel['id'] = range(len(hotel))
# hotel.to_csv("dataset\\dt\\hotelF.csv", index=False)
# from sklearn.cluster import MiniBatchKMeans
# minibatch_kmeans = MiniBatchKMeans(n_clusters=5)


kmeans =KMeans(n_clusters = 6 , n_init=10 , max_iter=300) #le meilleur nbr de cluster cest 6 dapres le graphe

kmeans.fit(coordinate)

centroids = kmeans.cluster_centers_

hotel['cluster_label'] = kmeans.labels_


def get_coordinates(city):
    geolocator = Nominatim(user_agent="recomendation_app")
    location = geolocator.geocode(city)
    if location:
        latitude = location.latitude
        longitude = location.longitude
        return latitude, longitude
    else:
        return None



def requirementbased(city, number, features):
    hotel['city'] = hotel['city'].str.lower()
    hotel['roomamenities'] = hotel['roomamenities'].str.lower()
    features = features.lower()
    features_tokens = word_tokenize(features)
    sw = stopwords.words('english')
    lemm = WordNetLemmatizer()
    f1_set = {w for w in features_tokens if not w in sw}
    f_set = set()

    coordinates=get_coordinates(city)
    lat=coordinates[0]
    long=coordinates[1]

    new_data = np.array([[lat, long]])
    
    cluster=kmeans.predict(new_data)

    h=hotel[hotel['cluster_label'] == cluster[0]]
    
    for se in f1_set:
        f_set.add(lemm.lemmatize(se))
    reqbased = h[h['city'] == city.lower()]
    reqbased = reqbased[reqbased['guests_no'] == number]
    reqbased = reqbased.set_index(np.arange(reqbased.shape[0]))

    l1 = []; l2 = []; cos = [];
    for i in range(reqbased.shape[0]):
        temp_tokens = word_tokenize(reqbased['roomamenities'][i])
        temp1_set = {w for w in temp_tokens if not w in sw}
        temp_set = set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        cos.append(len(rvector))
    reqbased['similarity'] = cos
    reqbased = reqbased.sort_values(by='starrating', ascending=False)
    reqbased = reqbased.sort_values(by='similarity', ascending=False)
    reqbased.drop_duplicates(subset='hotelcode', keep='first', inplace=True)
    m=reqbased[['id','hotelname','city', 'starrating', 'address', 'roomamenities','similarity','image']]
    return m

# print(requirementbased('paris',2,' '))

from math import sin, cos, sqrt, atan2, radians
R=6373.0#Earth's Radius
sw = stopwords.words('english')
lemm = WordNetLemmatizer()

def hybrid(city,number,features):
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

    h=hotel[hotel['cluster_label'] == cluster[0]]

    dist=[]
    
    lat1=radians(float(lat1))
    long1=radians(float(long1))
    hybridbase=h[h['guests_no']==number]
    hybridbase['city']=hybridbase['city'].str.lower()
    hybridbase=hybridbase[hybridbase['city']==city.lower()]
    hybridbase.drop_duplicates(subset='hotelcode',inplace=True,keep='first')
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
        temp_tokens=word_tokenize(hybridbase['roomamenities'][i])
        temp1_set={w for w in temp_tokens if not w in sw}
        temp_set=set()
        for se in temp1_set:
            temp_set.add(lemm.lemmatize(se))
        rvector = temp_set.intersection(f_set)
        coss.append(len(rvector))
    hybridbase['similarity']=coss
    x=hybridbase.sort_values(by='similarity',ascending=False).head(10)
    m=x[['id','hotelname','city', 'starrating', 'address', 'roomamenities','similarity','image','distance']]
    return m


print(hybrid('paris',2,'wifi'))

def gethotel(id):
    x = hotel[hotel['id'] == id]
    return x


# inertia = kmeans.inertia_
# print("Inertia: ", inertia)

# # Calculate the silhouette score
# labels = kmeans.labels_
# silhouette_avg = silhouette_score(coordinate, labels)
# print("Silhouette Score: ", silhouette_avg)

# print(gethotel(10))

# train_labels = kmeans.labels_

# Calculate the inertia
# inertia = kmeans.inertia_
# plt.scatter(hotel.iloc[:, 0], hotel.iloc[:, 1], c=train_labels)
# plt.xlabel('Column 1')
# plt.ylabel('Column 2')
# plt.title('K-means Clustering')
# plt.show()

# dataset=hotel
# # Extraction des coordonnées de latitude et de longitude
# latitude = dataset['latitude']
# longitude = dataset['longitude']

# # Affichage des points sur la carte
# plt.figure(figsize=(10, 6))
# plt.scatter(longitude, latitude, c='b', alpha=0.5)
# plt.xlabel('longitude')
# plt.ylabel('latitude')
# plt.title('Carte de la dataset')
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt

# # Supposons que vous avez une dataframe appelée "df" contenant votre dataset

# # Visualisation des données
# plt.figure(figsize=(10, 6))
# plt.scatter(hotel['id'],hotel['hotelname'],hotel['city'], hotel['starrating'], hotel['address'], hotel['roomamenities'],hotel['image'])  # Remplacez 'feature1' et 'feature2' par les noms de vos colonnes
# plt.title('Visualization of Dataset')

# # Affichage de la figure
# plt.show()
