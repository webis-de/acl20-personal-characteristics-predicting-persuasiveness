"""
- create author-topcategory dictionary-vectors from author-subreddit
- use categories parsed from snoopsnoo
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
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))
    local_output_root = kwargs.get('local-output-root', None)
    subreddit_category_pickle = kwargs['subreddit-category-pickle']

    _subreddit_topcategory = pickle.load(open('/home/username/data/output/_jobs/subreddit_topcategory.pickle','rb'))
    subreddit_topcategory = sc.broadcast(_subreddit_topcategory)

    author_subreddit = sc.pickleFile(input_path)

    author_category = author_subreddit.map(lambda x: subreddits_to_categories(x, subreddit_topcategory) )

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_category.saveAsPickleFile(output_path)


def subreddits_to_categories(author, subreddit_category):
    subreddits = author['subreddits']
    category_subreddits = {}
    for s in subreddits:
        name = s['subreddit']

        cat = subreddit_category.value.get(name, 'Other')

        count = s['submissions'] + s['comments']

        if cat not in category_subreddits:
            category_subreddits[cat] = {}
        category_subreddits[cat][name] = count

    author['topcategory_subreddits'] = category_subreddits

    return author
