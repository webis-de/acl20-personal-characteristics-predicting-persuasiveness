"""clean original dataset"""

from pyspark import SparkContext, SparkConf, SparkFiles
import sys
import json
import pprint
import csv
import time
import string
import re
import pickle
from collections import defaultdict
from functools import reduce
from itertools import groupby
from subprocess import Popen, PIPE
from jobs.shared import utils


def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    subs_path = kwargs['subs-path']
    coms_path = kwargs['coms-path']
    botlist_csv = kwargs['botlist-csv']

    with open(botlist_csv, 'r') as f:
        reader = csv.reader(f)
        botlist = set(list(map(lambda x: x[0], list(reader))))
    # botlist = sc.broadcast(_botlist)

    ### leave only authors from pairs
    authors_from_pairs = set(pickle.load(open('/home/username/data/output/_jobs/authors_from_pairs.pickle','rb')))

    subs_data = sc.textFile(subs_path).map(lambda x: jsonloads(x) )\
        .filter(lambda x: x is not '' and 'author' in x and 'selftext' in x\
            and x['author'] not in botlist and len(x['selftext']) > 0 and x['selftext'] not in ['', '[removed]', '[deleted]']\
            and x['author'] in authors_from_pairs)

    coms_data = sc.textFile(coms_path).map(lambda x: jsonloads(x) )\
        .filter(lambda x: x is not ''  and 'author' in x and 'body' in x\
            and x['author'] not in botlist and len(x['body']) > 0 and x['body'] not in ['', '[removed]', '[deleted]'] and x['author'] in authors_from_pairs)

    subs_data = subs_data.coalesce(250)
    coms_data = coms_data.coalesce(500)

    subs_data.saveAsPickleFile('/user/username/data/output/_jobs/cleaned_cmv_authors2/subs')
    coms_data.saveAsPickleFile('/user/username/data/output/_jobs/cleaned_cmv_authors2/coms')


def jsonloads(l):
    try:
        return json.loads(l)
    except:
        return ''

