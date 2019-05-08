from app import app
from flask import g, render_template, jsonify,request
from app.models import Post
from app import db
import logging
import calendar
import pandas as pd
import time
import socket
import os
from subprocess import call
import subprocess
import os
import numpy as np
import pickle
import csv  
import json
dirname = os.path.dirname(__file__)


tfidf_file = os.path.join(dirname, './csvs/tfidf_scores.csv')
sentiment_file=os.path.join(dirname, './csvs/sentiment_scores.csv')
dependency_file=os.path.join(dirname, './csvs/dependency_features_pca.csv')
char_bigram_file=os.path.join(dirname,'./csvs/char_bigram_features_pca.csv')
word_bigram_file=os.path.join(dirname,'./csvs/word_bigram_features_pca.csv')
tfidf_sparse_matrix_file=os.path.join(dirname,'./csvs/tfidf_features_pca.csv')
ensemble_file=os.path.join(dirname,'./ensemble-clf.sav')
stack_file=os.path.join(dirname,'./stack-clf.sav')
raw_data_file=os.path.join(dirname,'./raw_data.csv')
main_bat_file = os.path.join(dirname,'./main.bat')
pca_bat_fie = os.path.join(dirname,'./pca.bat')

toto_logger = logging.getLogger("toto")
toto_logger.setLevel(logging.DEBUG);
console_handler = logging.StreamHandler()
toto_logger.addHandler(console_handler)


# @app.route('/')
@app.route('/home/' , methods=['GET','POST'])
def home():
    query_string = request.args.get("post")
    body_string = request.args.get("body")
    
    host = socket.gethostbyname(socket.gethostname())
    print("host : "  , host);
    # toto_logger.debug("current search request: " + query_string);
    print("current search " , query_string)
    model(query_string);
    if(query_string!=None):
        addToQueue('Sas',query_string,body_string)
    return 'Hate Speech Server: added new post'

def addToQueue(username,post, body):
    currentTime =calendar.timegm(time.strptime('Jul 9, 2009 @ 20:02:58 UTC', '%b %d, %Y @ %H:%M:%S UTC'))
    p=Post(username=username,time=currentTime,post=post,body=body)
    db.session.add(p)
    db.session.commit()
#process the posts batchwise every 15 minutes
#
@app.route('/getAll' ,methods=['GET','POST'])
def getAll():
    allPosts = Post.query.all()
    all_post_arr=[]
    all_body_arr=[]
    for x in range(len(allPosts)):
        print("post: " ,allPosts[x].post)
        print("post body:" , allPosts[x].body)

        # indices.append(str(x))
        all_post_arr.append(allPosts[x].post)
        all_body_arr.append(allPosts[x].body)
    data = { 'question' : all_post_arr, 'body' : all_body_arr }
    # json_obj_q = json.dumps(all_post_arr)
    # json_obj_body = json.dumps(all_body_arr)
    json_obj = json.dumps(data)
    # print("json obj: " , json_obj_q)
    # print("json obj_2: " , json_obj_body)
    return json_obj


@app.route('/process' , methods=['GET','POST'])
def processQueue():
    all_post_arr=[]
    indices=[]
    allPosts = Post.query.all()
    if len(allPosts)!=0:
        for x in range(len(allPosts)):
            print("post: " ,allPosts[x].post)
            print("post body:" , allPosts[x].body)
            indices.append(str(x))
            all_post_arr.append(allPosts[x].post)
        df = pd.DataFrame({'tweet':all_post_arr})
        print("dataframe : " , df)
        df.to_csv(raw_data_file,index=False)
        # subprocess.call([r])
        
        subprocess.call([main_bat_file])
        subprocess.call([pca_bat_fie])
        predicted_values=predict()
        print("predicted values : " , predicted_values)
        hate_speech_arr=[]
        for i in range(len(predicted_values)):
            if(predicted_values[i]==0 or predicted_values[i]==1):
                hate_speech_arr.append(all_post_arr[i])
                print("hate speech detected in text: " , all_post_arr[i])
        json_obj = json.dumps(hate_speech_arr)
        return  json_obj
    else:
        return 'no posts to process..'

@app.route('/deleteall',methods=['GET'])
def deleteRows():
    db.session.query(Post).delete()
    db.session.commit()
    return 'deleting rows...'


def model(val):
    print ("running model...")


def predict():
    weighted_tfidf_score_op = pd.read_csv(tfidf_file,encoding='utf-8')
    sentiment_scores_op = pd.read_csv(sentiment_file,encoding='utf-8')
    dependency_features_op = pd.read_csv(dependency_file,encoding='utf-8')
    char_bigrams_op = pd.read_csv(char_bigram_file,encoding='utf-8')
    word_bigrams_op = pd.read_csv(word_bigram_file,encoding='utf-8')
    tfidf_sparse_matrix_op = pd.read_csv(tfidf_sparse_matrix_file,encoding='utf-8')

    df_list_op=[weighted_tfidf_score_op,sentiment_scores_op, dependency_features_op, char_bigrams_op, word_bigrams_op, tfidf_sparse_matrix_op]
    otpt = df_list_op[0]

    for df_op in df_list_op[1:]:
        otpt = otpt.merge(df_op , on='index')
    otpt = otpt.iloc[:,1:]
    otpt=otpt.dropna(how='any')
    # ensemble = pickle.load(open('ensemble-clf.sav', 'rb'))
    stack = pickle.load(open(stack_file, 'rb'))
    pred_op = stack.predict(otpt.values)

    print("Predicted values:" ,pred_op)
    # df_list_op=[weighted_tfidf_scorex_op,sentiment_scores_op, dependency_features_op, char_bigrams_op, word_bigrams_op, tfidf_sparse_matrix_op]
    # otpt = pd.concat(df_list_op,axis=1)
    # otpt=otpt.drop('index',1)
    # otpt=otpt.drop('Unnamed: 0',1)

    # otpt=otpt.dropna(how='any')

    # print("need to predict values on : " , otpt)

    # otpt.to_csv('otpt.csv',index=False)
    # # otpt = otpt.iloc[:,1:]
    # model = pickle.load(open(ensemble_file, 'rb'))
    # # 0 ->  hate speech 1 -> offensive langauge 2 -> neither
    # # model = pickle.load(open(stack_file, 'rb'))
    # pred_op = model.predict(otpt)
    return pred_op
    
