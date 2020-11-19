"""
- create author-category dictionary-vectors from author-subreddit
- use categories parsed from snoopsnoo
- categories not present in snoopsnoo belong to 'Others'
"""
from pyspark import SparkContext, SparkConf, SparkFiles
import json
import csv
import time
import pickle
from jobs.shared import utils

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None)) # /user/username/data/output/_jobs/author_subreddit/latest
    local_output_root = kwargs.get('local-output-root', None)
    subreddit_category_pickle = kwargs['subreddit-category-pickle']

    subreddit_category = pickle.load(open(subreddit_category_pickle, 'rb'))

    author_subreddit = sc.pickleFile(input_path) # /user/username/data/output/author_subreddit_vec/latest

    author_category = author_subreddit.map(lambda x: subreddits_to_categories(x, subreddit_category) )

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_category.saveAsPickleFile(output_path)


def subreddits_to_categories(author, subreddit_category):
    # subreddits_by_category
    subreddits = author['subreddits']
    categories = {}  # author-category vector containing post frequencies for each category
    category_subreddits = {}
    for s in subreddits:
        name = s['subreddit']

        cat = subreddit_category.get(name, 'Other')

        count = s['submissions'] + s['comments']

        if cat not in category_subreddits:
            category_subreddits[cat] = {}
        category_subreddits[cat][name] = count

        if cat not in categories:
            categories[cat] = count
        else:
            categories[cat] += count

    author['categories'] = categories
    author['category_subreddits'] = category_subreddits

    return author
