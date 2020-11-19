"""
- calculate features for author-subreddit vectors:
- freq (sub+com?), normfreq, bin (one-hot), tf-idf
"""

from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
from functools import reduce
import pprint
import math
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize


def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs.get('local-output-root', None)
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))
    subreddit_df_pickle = kwargs['subreddit-df-pickle']

    _subreddit_df = pickle.load(open(subreddit_df_pickle, 'rb'))
    subreddit_df = sc.broadcast(_subreddit_df)

    _dv = DictVectorizer()
    features_vec = _dv.fit_transform(_subreddit_df)
    dv = sc.broadcast(_dv)

    file = sc.textFile(input_path)
    data = file.map(lambda l: json.loads(l) )

    authors_total = data.count()
    author_subreddit_features = data.map(lambda x: get_feature_vectors(x, authors_total, subreddit_df, dv))

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name) # output_folder='/tf_squared'
    author_subreddit_features.saveAsPickleFile(output_path)


def get_feature_vectors(author, authors_total, subreddit_df, dv):
    subreddits = author['subreddits']

    all_terms_in_doc = reduce(lambda a, b: a + b, map(lambda x: (x['submissions'] + x['comments']), subreddits))    

    tfidf = {}
    lognorm = {}
    raw = {}
    for s in subreddits:
        try:
            k = s['subreddit']
            count = s['submissions'] + s['comments']
            idf = (math.log(authors_total / subreddit_df.value[k]))
            tfidf[k] = (count / all_terms_in_doc) * idf
            lognorm[k] = math.log(1 + count) * idf
            raw[k] = count
        except KeyError:
            print(k)

    author['sr_vec'] = {}
    author['sr_vec']['tfidf'] = dv.value.transform(tfidf)
    author['sr_vec']['lntfidf'] = dv.value.transform(lognorm)
    author['sr_vec']['raw'] = dv.value.transform(raw)
    author['sr_vec']['rawnorm'] = normalize(author['sr_vec']['raw'])

    return author

