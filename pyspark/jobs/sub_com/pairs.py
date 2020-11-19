'''
- generate pairs:
  - consider only threads with OPs replies (with as well as without delta)

For each comment:
0. Current comment is terminal: quit.
1. NOT terminal and there IS a reply by OP down the thread:
1.1 There IS direct comment by OPs
    1.1.1 if there IS NO interplay OR (there IS interplay AND current comment IS delta):
        - consider current comment as pair (add current comment and "context"(!) 
        to the list of comments by authors within the thread). 
        Check for child comment by DeltaBot to mark with ∆.
1.2 There IS NO direct comment by OP. Iterate over children: go to step 0.
2. There IS NO OPs comment down the thread: quit.
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
from subprocess import Popen, PIPE
from jobs.shared import utils

pp = pprint.PrettyPrinter(width=100)

def analyze(sc, **kwargs):
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))
    botlist_csv = kwargs['botlist-csv']


    with open(botlist_csv, 'r') as f:
        reader = csv.reader(f)
        _botlist = list(map(lambda x: x[0], list(reader)))
    botlist = sc.broadcast(_botlist)

    print("input_path: " + input_path)
    file = sc.textFile(input_path)
    threads = file.map(lambda l: json.loads(l))
    pairs = threads.flatMap(lambda x: top_level(x))
    pairs = pairs.filter(lambda x: x[0]['author'] not in botlist.value and x[1]['author'] not in botlist.value)

    ### - filter out pairs where sub['selftext'] or com['body'] empty
    ### - OR author == '[removed]/[deleted]'

    pairs_json = pairs.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    pairs_json.saveAsTextFile(output_path)

def top_level(sub):
    pairs = []
    top_coms = sub['comments']
    op = sub['author']

    for com in top_coms:
        context = defaultdict(list)
        generate_pairs(com, sub, op, pairs, context)

    reduced = reduce_by_author(pairs)

    return reduced

def generate_pairs(comment, sub, op, pairs, context):
    if is_terminal(comment) or not is_ops_reply(comment, op):
        return pairs
    author = comment['author']
    interplay = is_interplay(comment, author, op)
    delta = is_delta(comment, op)
    direct_op_reply = is_direct_op_reply(comment, op)
    if direct_op_reply and (not interplay or (interplay and delta)):
        c = {k:v for k,v in comment.items() if k not in ['children']}
        c['delta'] = delta
        c['context'] = copy.copy(context[author])
        s = {k:v for k,v in sub.items() if k not in ['comments']}
        context[op].append({k:v for k,v in direct_op_reply.items() if k not in ['children']})
        s['context'] = copy.copy(context[op])
        pairs.append( (s, c) )

    context[author].append({k:v for k,v in comment.items() if k not in ['children']})
    children = comment['children']
    for com in children:
        generate_pairs(com, sub, op, pairs, context)

    return pairs

### leave only positives or one negative pair for each comment author within a submission
def reduce_by_author(pairs):
    new = []
    mapped = map(lambda x: (x[1]['author'], x), pairs)
    get_first = lambda p: p[0]
    get_second = lambda p: p[1]
    grouped = groupby(sorted(mapped, key=get_first), get_first)
    for g in grouped:
        neg = ()
        pos = []
        for x in g[1]:
            if x[1][1]['delta']:
                pos.append( (x[1][0], x[1][1]) )
            else:
                neg = (x[1][0], x[1][1])
        if len(pos):
            new.extend(pos)
        elif neg:
            new.append(neg)

    return new

def is_ops_reply(com, op):
    res = False
    for c in com['children']:
        if c['author'] == op:
            return True
        else:
            if is_ops_reply(c, op):
                return True
    return False

def is_direct_op_reply(com, op):
    for c in com['children']:
        if c['author'] == op:
            return c
    return False

def is_delta(com, op):
    for c in com['children']:
        for x in c['children']:
            if x['author'] == 'DeltaBot':
                return True
    return False

def is_terminal(com):
    return len(com['children']) == 0

def is_interplay(com, author, op):
    print(len(com['children']))
    for c in com['children']:
        if c['author'] == author:
            for x in c['children']:
                if x['author'] == op:
                    return True
        else:
            if is_interplay(c, author, op):
                return True
    return False
