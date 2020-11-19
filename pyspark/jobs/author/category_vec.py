"""vectorize author-category dictionaries"""

from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
from functools import reduce
import pprint
import math
import csv
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))
    category_df_pickle = kwargs['category-df-pickle']

    _category_df = pickle.load(open(local_output_root+'/category_df.pickle', 'rb'))
    category_df = sc.broadcast(_category_df)

    _dv = DictVectorizer()
    features_vec = _dv.fit_transform(_category_df)
    dv = sc.broadcast(_dv)

    data = sc.pickleFile(input_path) # /user/username/data/output/author_category/latest

    authors_total = data.count()
    author_category_features = data.map(lambda x: get_feature_vectors(x, authors_total, category_df, dv))

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_category_features.saveAsPickleFile(output_path)


def get_feature_vectors(author, authors_total, category_df, dv):
    categories = author['categories']

    all_terms_in_doc = reduce(lambda a,b: a + b, map(lambda x: x[1], categories.items()) )

    tfidf = {}
    lognorm = {}
    raw = {}
    for k,v in categories.items():
        try:
            count = v
            idf = (math.log(authors_total / category_df.value[k]))
            tfidf[k] = (count / all_terms_in_doc) * idf
            lognorm[k] = math.log(1 + count) * idf
            raw[k] = count
        except KeyError:
            print(k)

    author['cat_vecs'] = {}
    author['cat_vecs']['tfidf'] = dv.value.transform(tfidf)
    author['cat_vecs']['lntfidf'] = dv.value.transform(lognorm)
    author['cat_vecs']['raw'] = dv.value.transform(raw)
    author['cat_vecs']['rawnorm'] = normalize(author['cat_vecs']['raw'])

    return author
