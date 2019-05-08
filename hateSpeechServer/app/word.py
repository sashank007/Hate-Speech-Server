import pandas as pd
import re
import numpy as np
import csv
import os
dirname = os.path.dirname(__file__)
word_bigram_file=os.path.join(dirname, './csvs/word_bigram_features.csv')
word_bigram_pca_file=os.path.join(dirname, './csvs/word_bigram_features_pca.csv')
word_bigrams = pd.read_csv(word_bigram_file)

word_bigrams=word_bigrams.drop('index',1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(word_bigrams)
word_bigrams_scale = scaler.transform(word_bigrams)
# index_values = word_bigrams_scale.shape[0]
# indices=[i for i in range(0,index_values)]

# word_bigrams_scale['index']=indices
# print(word_bigrams_scale)
#Applying PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
word_bigrams_transform =pca.fit_transform(word_bigrams_scale)

df=pd.DataFrame(word_bigrams_transform,columns=['feature10', 'feature11', 'feature12'])
# .to_csv(word_bigram_pca_file)

# df= pd.DataFrame(char_bigrams_transform,columns=['feature1', 'feature2', 'feature3'])
df = df.reset_index();
df.to_csv(word_bigram_pca_file,index=False)

