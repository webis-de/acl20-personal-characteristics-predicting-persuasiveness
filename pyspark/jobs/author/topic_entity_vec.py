"""author-topic-entity vectors from author-entity"""

from pyspark import SparkContext, SparkConf, SparkFiles
import json
import csv
import time
import math
import pickle
from functools import reduce
from jobs.shared import utils
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))
    
    page_topics = pickle.load(open(local_output_root+'/page_topics.pickle', 'rb'))

    _entity_df = pickle.load(open(local_output_root+'/entity_df.pickle', 'rb'))
    entity_df = sc.broadcast(_entity_df)

    topics_df = {'Academic disciplines': 169126,
        'Arts': 165790,
        'Business': 165670,
        'Concepts': 159671,
        'Culture': 169696,
        'Education': 162128,
        'Entertainment': 166557,
        'Events': 157631,
        'Geography': 164197,
        'Health': 168352,
        'History': 166707,
        'Humanities': 169517,
        'Language': 168451,
        'Law': 163853,
        'Life': 167678,
        'Mathematics': 157341,
        'Nature': 167276,
        'Other': 129536,
        'People': 144695,
        'Philosophy': 163002,
        'Politics': 167504,
        'Reference': 157377,
        'Religion': 161830,
        'Science': 167156,
        'Society': 170080,
        'Sports': 158917,
        'Technology': 167069,
        'Universe': 160159,
        'World': 164604}

    topic_pages = pickle.load(open(local_output_root+'/topic_pages.pickle', 'rb'))
    # vectorizer for each topic
    _vectorizers = {}
    for k,v in topic_pages.items():
        dv = DictVectorizer()
        dv.fit_transform(v)
        _vectorizers[k] = dv
    vectorizers = sc.broadcast(_vectorizers)

    _dv = DictVectorizer()
    features_vec = _dv.fit_transform(topics_df)
    dv = sc.broadcast(_dv)

    data = sc.pickleFile(input_path)

    authors_total = data.count()
    author_topic_entity_vec = data.filter(lambda x: len(x['topics']) > 0 and len(x['topics_freq']) > 0)\
        .map(lambda x: get_feature_vectors(x, authors_total, entity_df, topics_df, vectorizers, dv) )

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_topic_entity_vec.saveAsPickleFile(output_path)

def get_feature_vectors(author, authors_total, entity_df, topics_df, vectorizers, dv):
    topics = author['topics']
    topics_freq = author['topics_freq']
    topics_pages = author['topics_pages']
    res = {}

    all_terms_in_doc = reduce(lambda a,b: a+b, map(lambda x: x[1], topics.items()))
    tp_vecs = {}
    for topic, pages in topics_pages.items():
        tfidf = {}
        lognorm = {}
        raw = {}
        for k,v in pages.items():
            try:
                count = v
                idf = (math.log(authors_total / entity_df.value[k]))
                tfidf[k] = (count / all_terms_in_doc) * idf
                lognorm[k] = math.log(1 + count) * idf
                raw[k] = count
            except KeyError:
                print(k)        

        tp_vecs[topic] = {}
        tp_vecs[topic]['vec_tfidf'] = vectorizers.value[topic].transform(tfidf)
        tp_vecs[topic]['vec_lntfidf'] = vectorizers.value[topic].transform(lognorm)
        tp_vecs[topic]['vec_raw'] = vectorizers.value[topic].transform(raw)
        tp_vecs[topic]['vec_rawnorm'] = normalize(tp_vecs[topic]['vec_raw'])

    res['author'] = author['author']
    res['tp_vecs'] = tp_vecs

    all_terms_in_doc = reduce(lambda a,b: a+b, map(lambda x: (x[1]), topics.items()))
    tfidf = {}
    lognorm = {}
    raw = {}
    for k,v in topics.items():
        try:
            count = v
            idf = (math.log(authors_total / topics_df[k]))
            tfidf[k] = (count / all_terms_in_doc) * idf
            lognorm[k] = math.log(1 + count) * idf
            raw[k] = count
        except KeyError:
            print(k)

    res['t_vecs'] = {}
    res['t_vecs']['tfidf'] = dv.value.transform(tfidf)
    res['t_vecs']['lntfidf'] = dv.value.transform(lognorm)
    res['t_vecs']['raw'] = dv.value.transform(raw)
    res['t_vecs']['rawnorm'] = normalize(res['t_vecs']['raw'])


    all_terms_in_doc = reduce(lambda a,b: a+b, map(lambda x: (x[1]), topics_freq.items()))
    tfidf = {}
    lognorm = {}
    raw = {}
    for k,v in topics_freq.items():
        try:
            count = v
            idf = (math.log(authors_total / topics_df[k]))
            tfidf[k] = (count / all_terms_in_doc) * idf
            lognorm[k] = math.log(1 + count) * idf
            raw[k] = count
        except KeyError:
            print(k)

    res['tf_vecs'] = {}
    res['tf_vecs']['tfidf'] = dv.value.transform(tfidf)
    res['tf_vecs']['lntfidf'] = dv.value.transform(lognorm)
    res['tf_vecs']['raw'] = dv.value.transform(raw)
    res['tf_vecs']['rawnorm'] = normalize(res['tf_vecs']['raw'])

    return res
