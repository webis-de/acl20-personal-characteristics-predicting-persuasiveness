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
    category_features_path = kwargs['category-features']
    category_df_path = kwargs['category-df']

    category_df = sc.textFile(category_df_path).map(lambda x: json.loads(x)).collect()
    _df_dict = {}
    for s in category_df:
        _df_dict.update({s['category']: s['df']})

    _dv = DictVectorizer()
    features_vec = _dv.fit_transform(_df_dict)
    dv = sc.broadcast(_dv)

    _category_features = sc.textFile(category_features_path).map(lambda x: json.loads(x)).map(lambda x: {x['author']: x['category_tfidf']}).collect()
    _cf_dict = {}
    for a in _category_features:
        _cf_dict.update(a)
    cf = sc.broadcast(_cf_dict)

    sub_com_file = sc.textFile(sub_com_path)
    sub_com = sub_com_file.map(lambda l: json.loads(l) )

    output = sub_com.map(lambda x: process_pair(x, dv, cf)).collect()

    with open(local_output_root + '/category_tfidf_0.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['sim', 'delta'])
        for row in output:
            csv_out.writerow(row)


def process_pair(pair, dv, cf):
    author1 = pair[0]['author']
    author2 = pair[1]['author']
    delta = pair[1]['delta']

    cats1 = cf.value[author1]
    cats2 = cf.value[author2]

    vec1 = dv.value.transform(cats1)
    vec2 = dv.value.transform(cats2)

    sim = cosine_similarity(vec1, vec2)[0][0]    

    return (sim, delta)