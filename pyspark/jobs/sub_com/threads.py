"""Prepare dataset of submission-comments threads"""

from pyspark import SparkContext, SparkConf, SparkFiles
import sys
import json
import pprint
import csv
import time
from collections import defaultdict
from functools import reduce
from itertools import groupby

conf = SparkConf().setAppName("reddit")
sc = SparkContext(conf=conf)

pp = pprint.PrettyPrinter(width=100)

subs_path = "/corpora/corpora-thirdparty/corpora-reddit/corpus-submissions/"
coms_path = "/corpora/corpora-thirdparty/corpora-reddit/corpus-comments/"

subs_data = sc.textFile(subs_path).map(lambda l: json.loads(l))
coms_data = sc.textFile(coms_path).map(lambda l: json.loads(l))

### reduce duplicates take latest one by ('retrieved_on': 1504556955)
coms_data = coms_data.map(lambda x: (x['id'], x)).reduceByKey(lambda a,b: a if a['retrieved_on'] > b['retrieved_on'] else b).map(lambda x: x[1])

### group comments by submission
coms_grouped = coms_data.map(lambda x: (x['link_id'], x)).groupByKey().mapValues(list)

### generate threads for each group of comments within submission: return tuples (link_id, [N threads of comments with top-level comments as parents])
coms_threads = coms_grouped.map(lambda x: generate_threads(x))

### Load submissions dataset and join with coms_threads
subs_data = subs_data.map(lambda x: update_sub(x))
subs_mapped = subs_data.map(lambda x: (x['id'], x))
sub_com = subs_mapped.join(coms_threads)

sub_com = sub_com.map(lambda x: merge_sub_coms(x))
sub_com = sub_com.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))

output_path = "hdfs://hadoop:8020/user/username/data/tmp/threads"
sub_com.saveAsTextFile(output_path)


def generate_threads(x):
    threads = []
    link_id = x[0]
    comments = x[1]

    ### take top-level comments
    grouped_by_parent = group_coms_by_parent(comments)
    try:
        top_coms = grouped_by_parent[link_id]
    except KeyError:
        top_coms = []

    for com in top_coms:
        com['level'] = 0
        com = get_sub_tree(com, grouped_by_parent, 0)

    return (link_id, top_coms)

def update_sub(x):
    x['id'] = "t3_"+x['id'] if not x['id'].startswith('t3_') else x['id']
    x['comments'] = []
    x['delta'] = True if x['link_flair_css_class'] == 'OPdelta' else False
    return x

def set_delta_com(x):
    for c in x[1]:
        if 'delta' not in c:
            c['delta'] = False
        if c['author'] == 'DeltaBot':
            parent_id = c['parent_id'][3:] if c['parent_id'].startswith('t1_') else c['id']
            ops = list(filter(lambda y: True if y['id'] == parent_id else False, x[1]))
            if len(ops) > 0:
                op = ops[0]
                parent_id2 = op['parent_id'][3:] if op['parent_id'].startswith('t1_') else op['id']
                coms = list(filter(lambda z: True if z['id'] == parent_id2 else False, x[1]))
                if len(coms) > 0:
                    com = coms[0]
                    com['delta'] = True

    return x

def get_sub_tree(com, grouped_by_parent, level):
    # if level >= 35:
    #     print('link_id', com['link_id'], 'parent_id', com['parent_id'], 'id', com['id'])
    level += 1

    com['children'] = []
    if 't1_'+com['id'] in grouped_by_parent:
        com['children'] = grouped_by_parent['t1_'+com['id']]
        for c in com['children']:
            c['level'] = level
            c = get_sub_tree(c, grouped_by_parent, level)
    return com

def group_coms_by_parent(comments):
    key_func = lambda x: x[0]
    d = {}
    mapped = list(map(lambda c: (c['parent_id'], c), comments))
    for k, v in groupby(sorted(mapped, key=key_func), key_func):
        l = []
        for c in list(v):
            l.append(c[1])
        d[k] = l
    return d

def merge_sub_coms(x):
    new = x[1][0]
    new['comments'] = x[1][1]
    return new
