from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
import json
import os
dirname = os.path.dirname(__file__)
dep_dict_file=os.path.join(dirname, '../csvs/dependency_dict.json')
data_file=os.path.join(dirname, '../csvs/cleaned_tweets.csv')
stan_file=os.path.join(dirname, './stanford-corenlp-full-2018-10-05')

print("stanford nlp running...")
nlp = StanfordCoreNLP(stan_file)
data=pd.read_csv(data_file)
print("data: " , data)
#data=data.loc[1:11,['index','tweet']]
new_dict = dict()

for index, row in data.iterrows():
    tweet = str(row['tweet'])
    idx = str(row['index'])
    new_dict[idx]=nlp.dependency_parse(tweet)

json = json.dumps(new_dict)
f = open(dep_dict_file,"w")
f.write(json)

f.close()
nlp.close()
