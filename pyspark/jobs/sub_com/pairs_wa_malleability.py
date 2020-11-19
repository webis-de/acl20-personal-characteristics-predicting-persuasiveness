'''
Preprocess data for Malleability task
- Submissions with > 10 comments
- at least one reply by OP
'''

from pyspark import SparkContext, SparkConf, SparkFiles
from collections import defaultdict
import sys
import json
import pprint
import csv
import time
import logging
from collections import defaultdict
from functools import reduce
from itertools import groupby
import copy
import string
import re
import pickle
from subprocess import Popen, PIPE
from jobs.shared import utils

pp = pprint.PrettyPrinter(width=100)

def analyze(sc, **kwargs):
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))
    botlist_csv = kwargs['botlist-csv']

    sc.addFile("hdfs://hadoop:8020/user/username/nltk_data", recursive = True)
    import nltk

    with open(botlist_csv, 'r') as f:
        reader = csv.reader(f)
        _botlist = list(map(lambda x: x[0], list(reader)))
    botlist = sc.broadcast(_botlist)

    print("input_path: " + input_path)
    file = sc.textFile(input_path) # /user/username/data/output/sub_com_threads
    threads = file.map(lambda l: json.loads(l))

    pairs = threads.map(lambda x: top_level(x, botlist)).filter(lambda x: x)

    pickle.dump(pairs.collect(), open('/home/username/data/output/_jobs/pairs_winargs_malleability.pickle','wb'))

    pairs_json = pairs.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    pairs_json.saveAsTextFile(output_path)

### Generate "rooted-path-units": top-level comment + interplay comments by top-level comment author
def top_level(sub, botlist):
    op = sub['author']
    s = {k:v for k,v in sub.items() if k not in ['comments']}

    ### filter top-level comments
    ### after all conditions there should be more than 10 top-level comments, otherwise skip
    ### filter out top-level comments by bots
    top_coms = list(filter(lambda com: com['author'] != '[deleted]' and com['body'] not in ['', '[removed]'] 
        and com['author'] not in set(botlist.value), sub['comments']))
    if len(top_coms) < 10 or op == '[deleted]' or sub['selftext'] in ['', '[removed]', '[deleted]'] or len(sub['selftext'].split(' ')) < 5:
        return {}

    ### remove footer from submission.selftext if present
    if "_____" in s['selftext'] and "Hello, users of CMV!" in s['selftext'] and not s['selftext'].startswith("_____"):
        s['selftext'] = s['selftext'].split("_____")[0]
    if "This is a footnote from the CMV moderators." in s['selftext']:
        s['selftext'] = s['selftext'].split("This is a footnote from the CMV moderators.")[0]

    delta_submission = False
    subtrees = []
    for com in top_coms:
        root_author = com['author']
        comments = []
        delta_subtree = False
        process_subtree(com, sub, op, root_author, comments)
        ### cut all comments after delta
        cut = []
        for c in comments:
            cut.append(c)
            if c['delta']:
                delta_subtree = True
                delta_submission = True
                break
        subtrees.append({'delta': delta_subtree, 'author': root_author, 'comments': cut})

    sub['subtrees'] = subtrees

    return sub


def process_subtree(comment, sub, op, root_author, comments):
    if comment['author'] == root_author:
        delta = is_delta(comment, op)
        c = {k:v for k,v in comment.items() if k not in ['children']}
        c['delta'] = delta
        comments.append(c)

    for com in comment['children']:
        process_subtree(com, sub, op, root_author, comments)

def is_delta(com, op):
    for c in com['children']:
        # consider only deltas given by OP
        if c['author'] == op:
            for x in c['children']:
                if x['author'] == 'DeltaBot':
                    return True
    return False

