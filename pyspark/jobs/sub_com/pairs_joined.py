"""
- join author features together
- add features to pairs
- generate csv: similarities of pairs (author-category-entity feature vectors)
"""

from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
import pickle
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

    author_subreddit_vec = sc.pickleFile('/user/username/data/output/_jobs/author_subreddit_vec/latest')
    author_subreddit_vec_mapped = author_subreddit_vec.map(lambda x: (x['author'], {'sr_vec': x['sr_vec']}) )
    author_catsr_vec = sc.pickleFile('/user/username/data/output/_jobs/author_category_subreddit_vec/latest')
    author_catsr_vec_mapped = author_catsr_vec.map(lambda x: (x['author'], {'cat_vecs': x['cat_vecs'], 'catsr_vecs': x['catsr_vecs']}) )
    author_entity_vec = sc.pickleFile('/user/username/data/output/_jobs/author_entity_vec/latest')
    author_entity_vec_mapped = author_entity_vec.\
        map(lambda x: (x['author'], {'ent_vecs': x['ent_vecs']}) )

    tmp0 = author_subreddit_vec_mapped.join(author_catsr_vec_mapped)
    tmp1 = tmp0.map(lambda x: (x[0], {'cat_vecs': x[1][1]['cat_vecs'], 'catsr_vecs': x[1][1]['catsr_vecs'], 'sr_vecs': x[1][0]['sr_vec'] } ) )
    tmp2 = tmp1.join(author_entity_vec_mapped)
    author_features = tmp2.map(lambda x: (x[0], {'cat_vecs': x[1][0]['cat_vecs'], 'catsr_vecs': x[1][0]['catsr_vecs'], 'sr_vecs': x[1][0]['sr_vecs'], 'ent_vecs': x[1][1]['ent_vecs'] } ) )
    author_features.take(1)


    pairs_rdd = sc.textFile('/user/username/data/output/_jobs/sub_com_pairs/latest', use_unicode=False)\
        .map(lambda l : json.loads(l.decode("utf-8")) ).map(lambda x: (x[0]['author'], x[1]['author'], x[1]['delta']) )
    pairs_rdd.take(1)

    pairs_joined = pairs_rdd\
        .map(lambda x: ( x[0], {'author1': {'name': x[0]}, 'author2': {'name': x[1]}, 'delta': x[2]} ))\
        .join(author_features)\
        .map(lambda x: (x[1][0]['author2']['name'], update_dict(x[1][0], 'author1', x[1][1]) ))\
        .join(author_features)\
        .map(lambda x: update_dict(x[1][0], 'author2', x[1][1]) )

    pairs_joined.saveAsPickleFile('/user/username/data/output/_jobs/pairs_features')

def update_dict(d, a, features):
    d[a]['features'] = features
    return d