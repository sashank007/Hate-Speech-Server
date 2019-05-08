import pandas as pd
import re
import numpy as np
import csv
import os
dirname = os.path.dirname(__file__)
tfidf_file=os.path.join(dirname, './csvs/tfidf_features.csv')
tfidf__pca_file=os.path.join(dirname, './csvs/tfidf_features_pca.csv')
tfidf_features = pd.read_csv(tfidf_file)

# tfidf_features=tfidf_features.drop('index',1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(tfidf_features.iloc[:,1:])
tfidf_features_scale = scaler.transform(tfidf_features.iloc[:,1:])


#Applying PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
tfidf_features_transform =pca.fit_transform(tfidf_features_scale)

df  = pd.DataFrame(tfidf_features_transform,columns=['feature7', 'feature8', 'feature9'])
# .to_csv(tfidf__pca_file)


# df= pd.DataFrame(char_bigrams_transform,columns=['feature1', 'feature2', 'feature3'])
df = df.reset_index();
df.to_csv(tfidf__pca_file,index=False)
