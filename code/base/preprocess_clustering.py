import pandas as pd
import numpy as np
import pickle
import re
import math
import string
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from gensim.models import FastText
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from .plot import Plots

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer 
from nltk.tokenize import RegexpTokenizer


class Preprocess_Clustering:
    def __init__(self):
        self.stopwords = stopwords.words('english')
        self.stemmer = SnowballStemmer('english')
        self.tokenizer = RegexpTokenizer(r'\w+')
        print("object created")
    
    
    def load_data(self, path):
        data = pd.read_csv(path, sep='\t', names=['Command', 'time'])
        data = data.loc[data.Command != '{"clock":1}', :]
        data.Command = data.Command.replace('\{|\}|"Command":|"id":|"Source":|"Destination":|"', "", regex = True)
        data = data.loc[((~(data.Command.str.contains('getTempData', regex= True, na=False))) & 
                                    (~(data.Command.str.contains('getBrightnessData', regex= True, na=False))))]
        data[['Command_v1', 'id', 'Source', 'Destination']] = data.Command.str.split(',', expand=True)
        data.drop(['Command'], axis=1, inplace=True)
        data['Command_v2'] = data.Command_v1.replace(":\d+\.*\d*", "", regex=True)
        data.Source.fillna('LightSensor', inplace = True)
        data.Destination.fillna('Controller', inplace = True)
        data['Time_integer'] = data.time.astype('int')
        
        return data
    
    def preprocess_Command(self, data, stem = False):
        sentence = [data]
        text = []
        for word in sentence:
            
            if word not in self.stopwords:
                
                if stem:
                    text.append(self.stemmer.stem(word).lower())
                else:
                    text.append(word.lower())
        return self.tokenizer.tokenize(" ".join(text))
    
    def prepration_data(self, data, end, group_divider):
        df = data.loc[(data['Time_integer'] <= end), :].groupby(by='Time_integer').agg({'Command_v2': ' '.join}).reset_index()
        df = df.groupby(df.index // group_divider).agg({'Command_v2': ' '.join}).reset_index()
        df['tokenized_sents'] = df.Command_v2.map(self.preprocess_Command)
        
        return df
    
    
    def filter_data(self, data, column, filter_index):
      start_index = data.index[data.column == filter_index].tolist()[0]
      filtered_df = data.loc[:start_index-1, :] 
      
      return filtered_df 
    
    def extract_numerical_command(self, data,  column, filter_index):
        filtered_df = self.filter_data(data, column, filter_index)
        filtered_df = filtered_df.dropna()
        num_dataframes = filtered_df.id.unique()
        print("The unique ids are:", num_dataframes)
        
        dataframes = {}
        for i in num_dataframes:
            dataframes[f'df_{i}'] = filtered_df.loc[filtered_df.id == i, :]
            
        return dataframes
        
        
    
    def word2vec(self, command, vector_size, window, epochs, workers=4):
        model = Word2Vec(command, vector_size=vector_size, window=window, epochs=epochs, workers=workers)

        # getting the embedding vectors
        vocab = list(model.wv.key_to_index)
        X = model.wv[vocab]
        
        return(X, vocab, model)

    def fastText(self, command, vector_size, window, epochs, sg= 1, workers = 4, seed=42):
        model = FastText(command, vector_size=vector_size, window=window, epochs = epochs, sg = sg, workers=workers, seed = seed)

        # getting the embedding vectors
        vocab = list(model.wv.key_to_index)
        X = model.wv[vocab]
        
        return(X, vocab, model)

    def save_model(self, model, path):
        pickle.dump(model, open(path, 'wb'))
        print("Your model saved")
        
    
    def load_model(self, path):
        print("your model is loading")
        loaded_model = pickle.load(open(path, "rb"))
        
        return loaded_model
        
    
    def normalized_output(self, output):
        scalar = StandardScaler()
        X_normalized = scalar.fit_transform(output)
        
        return X_normalized
    
    def PCA(self, n_components, alg_output):
        # dimentionality reduction using PCA
        pca = PCA(n_components=n_components)
        # running the transformations
        pca_result = pca.fit_transform(alg_output)
        pca_result_normalized = pca.fit_transform(self.normalized_output(alg_output))
        
        return(pca_result, pca_result_normalized)  
        
    def UMAP(self, n_components, alg_output):
        umap = UMAP(n_components=n_components, init='random', random_state=0)
        umap_result = umap.fit_transform(alg_output)
        umap_result_normalized = umap.fit_transform(self.normalized_output(alg_output))
        
        return(umap_result, umap_result_normalized)
    
    def DReduction_to_DataFrame(self, Dreduction_output, vocab, dimension_number, label=None):
        df = pd.DataFrame(Dreduction_output, columns=[char for char in string.ascii_uppercase[:dimension_number]])
        df['Words'] = vocab
        # converting the lower case text to title case
        df['Words'] = df['Words'].str.title()
        df['label'] = label
        
        return df
            
    def kmeans(self, data, n_cluster):
        kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(data)
        kmeans_label = kmeans.labels_
        
        return kmeans_label
    
    def SpectralClustering(self, data, n_cluster, assign_labels = 'kmeans'):
        Spectral = SpectralClustering(assign_labels=assign_labels, n_clusters=n_cluster, random_state=0).fit(data)
        Spectral_label = Spectral.labels_
        
        return Spectral.labels_
        
    def DBSACN(self, data, eps, min_samples):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
        dbscan_label = dbscan.labels_
        return dbscan.labels_

        
    def AgglomerativeClustering(self, data, n_clusters):
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        predicted_labels = agg_clustering.fit_predict(data)
        
        return predicted_labels