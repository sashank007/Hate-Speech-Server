import pandas as pd
import re
import numpy as np
import csv
import os
dirname = os.path.dirname(__file__)
char_bigrams_file=os.path.join(dirname, '../csvs/char_bigram_features.csv')
char_bigrams_pca_file=os.path.join(dirname, '../csvs/char_bigram_features_pca.csv')
char_bigrams = pd.read_csv(char_bigrams_file)

# char_bigrams = char_bigrams.drop('index',1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(char_bigrams.iloc[:,1:])
char_bigrams_scale = scaler.transform(char_bigrams.iloc[:,1:])
 
#Applying PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
char_bigrams_transform =pca.fit_transform(char_bigrams_scale)

# char_bigrams_transform.drop
# pd.DataFrame(char_bigrams_transform,columns=['feature1', 'feature2', 'feature3']).to_csv(char_bigrams_pca_file,index=False)
df= pd.DataFrame(char_bigrams_transform,columns=['feature1', 'feature2', 'feature3'])
df = df.reset_index();
df.to_csv(char_bigrams_pca_file,index=False)

