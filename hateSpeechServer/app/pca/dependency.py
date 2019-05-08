import pandas as pd
import re
import numpy as np
import csv
import os
dirname = os.path.dirname(__file__)
dependency_file=os.path.join(dirname, './csvs/dependency_features.csv')
dependency_pca_file=os.path.join(dirname, './csvs/dependency_features_pca.csv')
dependency_features = pd.read_csv(dependency_file)

# dependency_features = dependency_features.drop('index',1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dependency_features.iloc[:,1:])
dependency_features_scale = scaler.transform(dependency_features.iloc[:,1:])


#Applying PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=3)
dependency_features_transform =pca.fit_transform(dependency_features_scale)

df=pd.DataFrame(dependency_features_transform,columns=['feature4', 'feature5', 'feature6'])
# .to_csv(dependency_pca_file)

# df= pd.DataFrame(char_bigrams_transform,columns=['feature1', 'feature2', 'feature3'])
df = df.reset_index();
df.to_csv(dependency_pca_file,index=False)
