import os
import glob
import re
import math
import nltk
nltk.download('punkt')
import string
from collections import defaultdict
import math
import pandas as pd
import csv
import os
dirname = os.path.dirname(__file__)
data_file=os.path.join(dirname, './csvs/cleaned_tweets.csv')
hatebase_file=os.path.join(dirname, './csvs/hatebase_dict.csv')
tfidf_file = os.path.join(dirname, './csvs/tfidf_scores.csv')
print("running tf-idf...")
data=pd.read_csv(data_file)
#data=data[1:11]
tweet = data['tweet']
index = data['index']

# dictionary into a list
dict1=pd.read_csv(hatebase_file)
dict2 = dict1['dic']
dic = []
for row in dict2:
    row = row.strip("',")
    dic.append(row)
#print(dic)


# Regular expression
def preprocess(text):
    # text = text.strip().strip('"')
    text = re.sub(r'[^A-Za-z0-9(),!?\.\'\`]', ' ', text)
    text = re.sub(r"\'s", " \'s", text)
    text = re.sub(r"\'ve", " \'ve", text)
    text = re.sub(r"n\'t", " n\'t", text)
    text = re.sub(r"\'re", " \'re", text)
    text = re.sub(r"\'d", " \'d", text)
    text = re.sub(r"\'ll", " \'ll", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", "  \. ", text)
    text = re.sub(r"\"", " , ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\(", " \( ", text)
    text = re.sub(r"\)", " \) ", text)
    text = re.sub(r"\?", " \? ", text)
    #text = re.sub(r"\S{2,}", " ", text)
    return text.strip().lower()

# calculate term frequency
def phrase_frequency(text):
    text = preprocess(text)
    tokens = nltk.word_tokenize(text)
    phrase1 = []
    for token in tokens:
        if not token.isalnum():
            continue
        else:
            phrase1.append(token)
    phrases = []
    for i in range(0,len(phrase1)-1):
        word=phrase1[i]+" "+phrase1[i+1]
        phrases.append(word)
    term_freqs = {}
    for phrase in phrases:
        if phrase in term_freqs:
            term_freqs[phrase] += 1 / len(phrases)
        else:
            term_freqs[phrase] = 1 / len(phrases)
    return term_freqs


def term_frequency(text):
    text = preprocess(text)
    tokens = nltk.word_tokenize(text)
    term_freqs = {}
    for token in tokens:
        if not token.isalnum():
            # filter out tokens that are not-alphanumeric, eg punctuation
            continue
        if token in term_freqs:
            term_freqs[token] += 1 / len(text)
        else:
            term_freqs[token] = 1 / len(text)
    return term_freqs

def term_fre(text):
    term_freqs = {}
    for word in term_frequency(text).keys():
        if word in dic:
            term_freqs[word] = term_frequency(text)[word]
        else:
            continue
    return term_freqs



docs = {}
for no, rows in enumerate(tweet,index[0]):
    docs[no] = term_fre(rows)
#print (docs[2])


# calculated document frequency
def document_frequency(docs):
    freqs = {}
    list1 = []
    print("docs: " , len(docs))
    for i in range(len(index)-1):
        for item in docs[i].keys():
            list1.append(item)
    # print("list1 :  " , list1)
    for term in list1:
        if term in freqs:
            freqs[term] += 1 / len(docs)
        else:
            freqs[term] = 1 / len(docs)
    return freqs
#print (document_frequency(docs))

# calculate tfidf score
def tfidf_score(docs):
    doc_freqs = document_frequency(docs)
    # print("length of docs: " , len(docs))
    print("doc_freqs: " , doc_freqs)
    for i in range(len(index)-1):
        for term in docs[i].keys():
            # print("doc_freq : "  , doc_freqs[term])   
            docs[i][term] = docs[i][term] * math.log(len(docs)/doc_freqs[term], 2)
    return docs
#print (tfidf_score(docs)[2])


score = tfidf_score(docs)
tot_score = {}
for i in range(len(index)-1):
    tot_score[i] = sum(score[i].values())



output = [['index','weighted_TFIDF_scores']]
for i in range(len(index)-1):
    temp = []
    temp.append(i)
    temp.append(tot_score[i])
    output.append(temp)

file=open(tfidf_file, 'w',newline='')
with file:
    writer = csv.writer(file)
    writer.writerows(output)
file.close()
pd.read_csv(tfidf_file,index_col = False)
