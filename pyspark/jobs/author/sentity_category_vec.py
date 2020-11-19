'''
- entity-sentiment feature vectors:
  - flatten author_entity
  - add (top-)categories
  - calculate and add median for each entity
  - group back by author
  - vectorize
  - save as (author: name, {category: vector})
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

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))

    _entity_df = pickle.load(open('/home/username/data/output/_jobs/entity_df.pickle', 'rb'))
    entity_df = sc.broadcast(_entity_df)

    _vectorizers_cats = pickle.load(open('/home/username/data/output/_jobs/vectorizers_cats.pickle','rb'))
    vectorizers_cats = sc.broadcast(_vectorizers_cats)
    _vectorizers_topcats = pickle.load(open('/home/username/data/output/_jobs/vectorizers_topcats.pickle','rb'))
    vectorizers_topcats = sc.broadcast(_vectorizers_topcats)

    _subreddit_category = pickle.load(open('/home/username/data/output/_jobs/subreddit_category_index.pickle','rb'))
    subreddit_category = sc.broadcast(_subreddit_category)
    _subreddit_topcategory = pickle.load(open('/home/username/data/output/_jobs/subreddit_topcategory.pickle','rb'))
    subreddit_topcategory = sc.broadcast(_subreddit_topcategory)

    author_entity = sc.pickleFile('/user/username/data/output/_jobs/author_entity/latest')

    flatten = author_entity.flatMap(lambda x: [(x[0],)+e for e in x[1]] )
    categories = flatten.map(lambda x: entity_to_categories(x, subreddit_category, subreddit_topcategory))
    # ('blackngold14', 'Personal foul (basketball)', 'Q15073191', 1100642, 'personal foul', -0.1779, 'Saints', 't3_2nupey') + (cat, topcat)

    ### map as ((subreddit, author, e_id), sent_score)
    ### map as ((cat/topcat, author, e_id), sent_score)
    medians_cat = categories.map(lambda x: ((x[8],x[0],x[1],x[3]),x[5]) ).groupByKey().mapValues(list).map(lambda x: x[0]+(float(np.median(x[1])),) )
    medians_topcat = categories.map(lambda x: ((x[9],x[0],x[1],x[3]),x[5]) ).groupByKey().mapValues(list).map(lambda x: x[0]+(float(np.median(x[1])),) )

    grouped_cat = medians_cat.groupBy(lambda x: x[1])
    grouped_topcat = medians_topcat.groupBy(lambda x: x[1])

    ### category_entities features vectors
    feature_vectors_cat = grouped_cat.map(lambda x: get_feature_vectors(x, entity_df, vectorizers_cats))
    feature_vectors_topcat = grouped_topcat.map(lambda x: get_feature_vectors(x, entity_df, vectorizers_topcats))

    feature_vectors_cat.saveAsPickleFile('/user/username/data/output/_jobs/sentity_category_vec')
    feature_vectors_topcat.saveAsPickleFile('/user/username/data/output/_jobs/sentity_topcategory_vec')


def entity_to_categories(x, subreddit_category, subreddit_topcategory):
    ### x[6] - subreddit
    return x + (subreddit_category.value.get(x[6],'Other'), subreddit_topcategory.value.get(x[6],'Other'))

def get_feature_vectors(author, entity_df, vectorizers):
    dict_cat = to_dict(author[1])

    _vecs = {}
    for cat, entities in dict_cat.items():
        _vecs[cat] = {}
        _vecs[cat] = vectorizers.value[cat].transform(entities)

    return {'author': author[0], 'sent_vec': _vecs}

def to_dict(entities_list):
    _dict = {}
    for e in entities_list:
        e_id = e[3]
        cat = e[0]
        score = e[4]
        if cat not in _dict:
            _dict[cat] = {}
        _dict[cat][e_id] = score
    return _dict


