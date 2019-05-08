import pandas as pd
import numpy as np
import pickle
import csv


weighted_tfidf_score_op = pd.read_csv('./csvs/tfidf_scores.csv',encoding='utf-8')
sentiment_scores_op = pd.read_csv('./csvs/sentiment_scores.csv',encoding='utf-8')
dependency_features_op = pd.read_csv('./csvs/dependency_features_pca.csv',encoding='utf-8')
char_bigrams_op = pd.read_csv('./csvs/char_bigram_features_pca.csv',encoding='utf-8')
word_bigrams_op = pd.read_csv('./csvs/word_bigram_features_pca.csv',encoding='utf-8')
tfidf_sparse_matrix_op = pd.read_csv('./csvs/tfidf_features_pca.csv',encoding='utf-8')

# weighted_tfidf_score_op = pd.read_csv('./csvs/tfidf_scores.csv',encoding='utf-8')
# sentiment_scores_op = pd.read_csv('./csvs/sentiment_scores.csv',encoding='utf-8')
# dependency_features_op = pd.read_csv('./csvs/dependency_features.csv',encoding='utf-8')
# char_bigrams_op = pd.read_csv('./csvs/char_bigram_features.csv',encoding='utf-8')
# word_bigrams_op = pd.read_csv('./csvs/word_bigram_features.csv',encoding='utf-8')
# tfidf_sparse_matrix_op = pd.read_csv('./csvs/tfidf_features.csv',encoding='utf-8')

df_list_op=[weighted_tfidf_score_op,sentiment_scores_op, dependency_features_op, char_bigrams_op, word_bigrams_op, tfidf_sparse_matrix_op]
otpt = df_list_op[0]

for df_op in df_list_op[1:]:
    otpt = otpt.merge(df_op , on='index')
otpt = otpt.iloc[:,1:]
# ensemble = pickle.load(open('ensemble-clf.sav', 'rb'))
stack = pickle.load(open('stack-clf.sav', 'rb'))
pred_op = stack.predict(otpt.values)

print("Predicted values:" ,pred_op)


# df_list_op=[weighted_tfidf_score_op,sentiment_scores_op, dependency_features_op, char_bigrams_op, word_bigrams_op, tfidf_sparse_matrix_op]
# # print(df_list_op)
# # otpt = df_list_op[0]

# # for df_op in df_list_op[1:]:
# #     otpt = otpt.merge(df_op , on='index')
# otpt = pd.concat(df_list_op,axis=1)
# otpt=otpt.drop('index',1)
# otpt=otpt.drop('Unnamed: 0',1)

# otpt=otpt.dropna(how='any')


# # df_list_op=[weighted_tfidf_score_op,sentiment_scores_op, dependency_features_op, char_bigrams_op, word_bigrams_op, tfidf_sparse_matrix_op]
# # otpt = df_list_op[0]

# # # for df_op in df_list_op[1:]:
# # #     otpt = otpt.merge(df_op , on='index')
# # otpt  = pd.concat(df_list_op , 1)
# # otpt=otpt.drop('index',1)
# # otpt= otpt.dropna(how='any')
# print("need to predict values on : " , otpt)

# otpt.to_csv('otpt.csv',index=False)
# # otpt = otpt.iloc[:,1:]
# # model = pickle.load(open('stack-clf.sav', 'rb'))
# # 0 ->  hate speech 1 -> offensive langauge 2 -> neither
# model = pickle.load(open('stack-clf.sav', 'rb'))
# pred_op = model.predict(otpt)
# print("Predicted values:" ,pred_op)
