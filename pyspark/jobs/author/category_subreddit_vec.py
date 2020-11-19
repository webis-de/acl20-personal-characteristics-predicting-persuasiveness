"""
- 'fine-grained' within category similarity: create author-category-subreddit dictionary-vectors from author-subreddit
- cats not found in snoopsnoo belong to 'Others'
"""

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
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None)) # /user/username/data/output/_jobs/author_subreddit/latest
    subreddit_category_pickle = kwargs['subreddit-category-pickle']
    subreddit_df_pickle = kwargs['subreddit-df-pickle']


    subreddit_category = pickle.load(open(subreddit_category_pickle, 'rb'))
    _subreddit_df = pickle.load(open(subreddit_df_pickle, 'rb'))
    subreddit_df = sc.broadcast(_subreddit_df)

    # create vectorizer for each category out of subreddit_df dataset
    subreddits_grouped_by_categories = subreddits_to_categories(_subreddit_df, subreddit_category)
    _vectorizers = {}
    for k,v in subreddits_grouped_by_categories.items():
        dv = DictVectorizer()
        dv.fit_transform(v)
        _vectorizers[k] = dv
    vectorizers = sc.broadcast(_vectorizers)


    data = sc.pickleFile(input_path) # /user/username/data/output/author_category_vec/latest

    authors_total = data.count()
    author_category = data.map(lambda x: get_feature_vectors(x, subreddit_df, authors_total, vectorizers) )

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_category.saveAsPickleFile(output_path)


def get_feature_vectors(author, subreddit_df, authors_total, vectorizers):
    category_subreddits = author['category_subreddits'] # subreddits grouped by category for 'fine-grained' similarities

    catsr_vecs = {}
    for cat, subreddits in category_subreddits.items():
        all_terms_in_doc = reduce(lambda a,b: a+b, map(lambda x: (x['submissions'] + x['comments']), author['subreddits']))
        tfidf = {}
        lognorm = {}
        raw = {}
        for k,v in subreddits.items():
            try:
                count = v
                idf = (math.log(authors_total / subreddit_df.value[k]))
                tfidf[k] = (count / all_terms_in_doc) * idf
                lognorm[k] = math.log(1 + count) * idf
                raw[k] = count
            except KeyError:
                print(k)        

        catsr_vecs[cat] = {}
        catsr_vecs[cat]['vec_tfidf'] = vectorizers.value[cat].transform(tfidf)
        catsr_vecs[cat]['vec_lntfidf'] = vectorizers.value[cat].transform(lognorm)
        catsr_vecs[cat]['vec_raw'] = vectorizers.value[cat].transform(raw)
        catsr_vecs[cat]['vec_rawnorm'] = normalize(catsr_vecs[cat]['vec_raw'])

    author['catsr_vecs'] = catsr_vecs

    return author

def subreddits_to_categories(subreddit_df, subreddit_category):
    # group subreddits by category
    category_subreddits = {}
    for k,v in subreddit_df.items():
        cat = subreddit_category.get(k, 'Other')

        if cat not in category_subreddits:
            category_subreddits[cat] = {}
        category_subreddits[cat][k] = v

    return category_subreddits

