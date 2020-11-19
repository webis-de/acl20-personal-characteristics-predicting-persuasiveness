"""generate csv: similarities of pairs (word2vec vectors author categories)"""

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
    snoopsnoo_path = kwargs['snoopsnoo']
    subreddit_df_path = kwargs['subreddit-df']

    # list used later to create row for csv
    subreddit_category_index = get_index(snoopsnoo_path)

    print('loading subreddit_df...')
    subreddit_df = sc.textFile(subreddit_df_path).map(lambda x: json.loads(x)).collect()
    subreddits_grouped_by_categories = subreddits_to_categories(subreddit_df, subreddit_category_index)

    all_categories_dict = {}
    for k, v in subreddits_grouped_by_categories.items():
        all_categories_dict[k] = 0
    print('subreddit_df loaded.')

    wv = pickle.load(open('/home/username/data/output/author_categories_wvectors', 'rb'))

    print('loading sub_com...')
    sub_com_file = sc.textFile(sub_com_path)
    sub_com = sub_com_file.map(lambda l: json.loads(l) ).collect()
    print('sub_com loaded.')

    print('processing pairs...')
    with open(local_output_root + '/category_name_w2v.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['sim', 'delta'])

        done = 0
        for pair in sub_com:
            done += 1
            if done % 10000 == 0:
                print(done)
            row = process_pair(pair, wv)
            if row is False:
                continue
            csv_out.writerow(row)


def get_full_row(row, all_categories_dict):
    tmp_cats = row[0]
    delta = row[1]

    cats = {}
    for k,v in all_categories_dict.items():
        cats[k] = tmp_cats[k] if k in tmp_cats else 0

    return [v for k,v in cats.items()] + [delta]

def process_pair(pair, wv):

    author1 = pair[0]['author']
    author2 = pair[1]['author']
    delta = pair[1]['delta']

    sub_vector = wv[author1]
    com_vector = wv[author2]

    cos_sim = cosine_similarity(sub_vector, com_vector)

    return (cos_sim[0][0], delta)

def common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)

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
