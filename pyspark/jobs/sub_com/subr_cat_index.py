"""generate subreddit-->category index as dict"""

from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
import pickle
from functools import reduce
import pprint
import math
import csv
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']
    snoopsnoo_path = kwargs['snoopsnoo']
    subreddit_df_path = kwargs['subreddit-df']

    subreddit_category_index = get_index(snoopsnoo_path)

    print('loading subreddit_df...')
    subreddit_df = sc.textFile(subreddit_df_path).map(lambda x: json.loads(x)).collect()
    subreddits_grouped_by_categories = subreddits_to_categories(subreddit_df, subreddit_category_index)

    all_categories_dict = {}
    for k, v in subreddits_grouped_by_categories.items():
        all_categories_dict[k] = 0
    print('subreddit_df loaded.')

    pickle.dump(open(local_output_root + 'subreddit_category_index'))


def subreddits_to_categories(subreddits, subreddit_category_index):
    categories_subreddits = {}

    for s in subreddits:
        name = s['subreddit']

        cat = subreddit_category_index.get(name, 'Other')

        count = s['df']

        if cat not in categories_subreddits:
            categories_subreddits[cat] = {}

        categories_subreddits[cat][name] = count

    return categories_subreddits

def get_index(file_path):
    with open(file_path) as f:
        data = json.load(f)

    ### index subreddit->category
    subreddit_category = {}
    for c in data['category']:
        for s in c['subreddits']:
            subreddit_category[s['name'][3:]] = c['name']
        if c.get('subcategory', None):
            for subc in c['subcategory']:
                for s in subc['subreddits']:
                    subreddit_category[s['name'][3:]] = subc['name']
                    ### 3d level categories
                    if subc.get('subcategory', None):
                        for subc2 in subc['subcategory']:
                            for s in subc2['subreddits']:
                                subreddit_category[s['name'][3:]] = subc2['name']

    return subreddit_category
