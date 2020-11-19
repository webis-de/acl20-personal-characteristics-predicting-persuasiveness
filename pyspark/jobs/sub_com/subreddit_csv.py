from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
from functools import reduce
import pprint
import math
import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']
    sub_com_path = kwargs['sub-com']
    subreddit_features_path = kwargs['subreddit-features']
    subreddit_df_path = kwargs['subreddit-df']
    botlist_csv = kwargs['botlist-csv'] # ../data/botlist.csv

    subreddit_df = sc.textFile(subreddit_df_path).map(lambda x: json.loads(x)).map(lambda x: {x['subreddit']: x['df']}).collect()
    _df_dict = {}
    for s in subreddit_df:
        _df_dict.update(s)

    _dv = DictVectorizer()
    features_vec = _dv.fit_transform(_df_dict)
    dv = sc.broadcast(_dv)

    _subreddit_features = sc.textFile(subreddit_features_path).map(lambda x: json.loads(x)).map(lambda x: {x['author']: x['tfidf']}).collect()
    _sf_dict = {}
    for a in _subreddit_features:
        _sf_dict.update(a)
    sf = sc.broadcast(_sf_dict)

    sub_com_file = sc.textFile(sub_com_path)
    sub_com = sub_com_file.map(lambda l: json.loads(l) )

    with open(botlist_csv, 'r') as f:
        reader = csv.reader(f)
        _botlist = list(map(lambda x: x[0], list(reader)))
    botlist = sc.broadcast(_botlist)
    sub_com = sub_com.filter(lambda x: x[0]['author'] not in botlist.value and x[1]['author'] not in botlist.value)

    '''
        - for each sub_com pair:
           - compute cosine similarity of author1 and author2
        - export to csv
    '''

    output = sub_com.map(lambda x: process_pair(x, dv, sf)).collect()

    with open(local_output_root + '/tfidf-tf-squared.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['sim', 'delta'])
        for row in output:
            csv_out.writerow(row)


def process_pair(pair, dv, sf):
    author1 = pair[0]['author']
    author2 = pair[1]['author']
    delta = pair[1]['delta']

    vec1 = dv.value.transform(sf.value[author1])
    vec2 = dv.value.transform(sf.value[author2])

    sim = cosine_similarity(vec1, vec2)[0][0]    

    return (sim, delta)