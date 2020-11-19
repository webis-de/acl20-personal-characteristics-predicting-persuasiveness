'''
- PCA on entity-frequency vectors:
  - flatten "entity-frequency feature vectors" as (author, category, vector)
  - group by category
  - stack sparse vectors within a category together
  - run PCA on precomputed category-entity-frequency feature vectors
  - map back as (author: name, {category: vector})
'''

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
from functools import reduce
from itertools import groupby
import numpy as np
import scipy
from sklearn.decomposition import PCA

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']


    author_entity_category_vec = sc.pickleFile('/user/username/data/output/_jobs/author_entity_category_vec/latest')

    flatten = author_entity_category_vec.flatMap(lambda x: [(x['author'], k, v) for k,v in x['topcatent_vec'].items()])
    grouped = flatten.groupBy(lambda x: x[1])#.mapValues(list)
    pcas = grouped.flatMap(lambda x: get_pca(x))
    grouped_by_author = pcas.groupBy(lambda x: x[0]).map(lambda x: {'author': x[0], 'topcatent_vec': to_dict(x)})

    grouped_by_author.saveAsPickleFile('/user/username/data/output/_jobs/entity_topcategory_pca')

def to_dict(group):
    _dict = {}
    for a,c,v in group[1]:
        _dict[c] = v
    return _dict

def get_pca(group):
    # group item: ('NegligibleSenescense', 'Cars', <1x14835 sparse matrix of type '<class 'numpy.float64'>' with 4 stored elements in Compressed Sparse Row format>)
    _index = []
    vecs = []
    for author, cat, vec in group[1]:
        _index.append(author)
        vecs.append(vec['tfidf'])
    stacked = scipy.sparse.vstack(vecs).toarray()
    n_samples = len(stacked)
    n_features = stacked[0].shape[0]
    print('cat', cat)
    print('n_samples',n_samples)
    print('n_features',n_features)
    n_components = min(min(n_samples, n_features),3)
    pca = PCA(n_components)
    res = pca.fit_transform(stacked)

    return [(i,cat,v) for i,v in zip(_index, res)]

def entity_to_categories(x, subreddit_category, subreddit_topcategory):
    ### x[7] - subreddit
    return x + (subreddit_category.value.get(x[6],'Other'), subreddit_topcategory.value.get(x[6],'Other'))


