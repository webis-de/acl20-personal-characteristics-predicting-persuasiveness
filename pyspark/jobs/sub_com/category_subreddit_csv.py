"""generate csv: similarities of pairs (author-category-subreddit feature vectors)"""

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
    sub_com_path = kwargs['sub-com']
    subreddit_category_pickle = kwargs['subreddit-category-pickle']
    subreddit_df_pickle = kwargs['subreddit-df-pickle']

    print('subreddit_df, grouped_by_cats...')
    subreddit_category_index = pickle.load(open(subreddit_category_pickle, 'rb'))
    subreddit_df = pickle.load(open(subreddit_df_pickle, 'rb'))
    subreddits_grouped_by_categories = subreddits_to_categories(subreddit_df, subreddit_category_index)

    print('all_categories_dict...')
    all_categories_dict = {}
    for k, v in subreddits_grouped_by_categories.items():
        all_categories_dict[k] = 0

    pairs_rdd = sc.textFile(sub_com_path, use_unicode=False)\
        .map(lambda l : json.loads(l.decode("utf-8")) ).map(lambda x: (x[0]['author'], x[1]['author'], x[1]['delta']) )    

    _csf = pickle.load(open('/home/username/data/output/_jobs/author_catsr_vec.pickle', 'rb'))
    csf = sc.broadcast(_csf)

    output = pairs_rdd.map(lambda x: process_pair(x, csf)).collect()

    print('processing pairs...')
    with open(local_output_root + '/pairs_author_catsr.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow([k for k, v in all_categories_dict.items()] + ['delta'])

        done = 0
        for pair in output:
            done += 1
            if done % 10000 == 0:
                print(done)
            csv_out.writerow(get_full_row(row, all_categories_dict))


def get_full_row(row, all_categories_dict):
    tmp_cats = row[0]
    delta = row[1]

    cats = {}
    for k,v in all_categories_dict.items():
        cats[k] = tmp_cats[k] if k in tmp_cats else 0

    return [v for k,v in cats.items()] + [delta]

def process_pair(pair, csf):

    author1 = pair[0]['author']
    author2 = pair[1]['author']
    delta = pair[1]['delta']

    sub_vectors = csf.value[author1]
    com_vectors = csf.value[author2]

    common = common_entries(sub_vectors, com_vectors)
    tmp_cats = {}
    for c in common:
        vec1 = c[1]
        vec2 = c[2]
        cos_sim = cosine_similarity(vec1, vec2)
        tmp_cats[c[0]] = cos_sim[0][0]

    if len(tmp_cats) == 0:
        return False

    return (tmp_cats, delta)

def common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

def subreddits_to_categories(subreddit_df, subreddit_category):
    category_subreddits = {}
    for k,v in subreddit_df.items():
        cat = subreddit_category.get(k, 'Other')

        if cat not in category_subreddits:
            category_subreddits[cat] = {}
        category_subreddits[cat][k] = v

    return category_subreddits

