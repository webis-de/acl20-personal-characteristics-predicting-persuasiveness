"""vectorize author-entity dictionaries"""

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
    entity_df_pickle = kwargs['entity-df-pickle']

    _entity_df = pickle.load(open(local_output_root+'/entity_df.pickle', 'rb'))
    entity_df = sc.broadcast(_entity_df)

    _dv = DictVectorizer()
    features_vec = _dv.fit_transform(_entity_df)
    dv = sc.broadcast(_dv)

    data = sc.pickleFile('/user/username/data/output/_jobs/author_entity_freq')

    authors_total = data.count()
    author_entity_features = data.map(lambda x: get_feature_vectors(x, authors_total, entity_df, dv))

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_entity_features.saveAsPickleFile(output_path)


def get_feature_vectors(author, authors_total, entity_df, dv):
    entities = author[1]

    all_terms_in_doc = reduce(lambda a,b: a + b, map(lambda x: x[1], entities) )

    tfidf = {}
    lognorm = {}
    raw = {}
    for v in entities:
        try:
            count = v[1]
            k = v[0][2]
            idf = (math.log(authors_total / entity_df.value[k]))
            tfidf[k] = (count / all_terms_in_doc) * idf
            lognorm[k] = math.log(1 + count) * idf
            raw[k] = count
        except KeyError:
            print(k)

    res = {}
    res['author'] = author[0]
    res['entities'] = author[1]
    res['ent_vecs'] = {}
    res['ent_vecs']['tfidf'] = dv.value.transform(tfidf)
    res['ent_vecs']['lntfidf'] = dv.value.transform(lognorm)
    res['ent_vecs']['raw'] = dv.value.transform(raw)
    res['ent_vecs']['rawnorm'] = normalize(res['ent_vecs']['raw'])

    return res

