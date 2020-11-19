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
    subs_path = kwargs['subs-path']
    coms_path = kwargs['coms-path']
    botlist_csv = kwargs['botlist-csv']

    with open(botlist_csv, 'r') as f:
        reader = csv.reader(f)
        _botlist = list(map(lambda x: x[0], list(reader)))
    botlist = sc.broadcast(_botlist)


    ### author-submissions frequencies
    subs_data = sc.pickleFile(subs_path) # corpora-reddit/corpus-submissions
    subs_data = subs_data.filter(lambda x: True if all (k in x for k in ('author', 'subreddit')) else False)
    # leave only authors from pairs
    authors_from_pairs = set(pickle.load(open('/home/username/data/output/_jobs/authors_from_pairs.pickle','rb')))
    subs_data = subs_data.filter(lambda x: x['author'] in authors_from_pairs)

    author_subreddit_submission = subs_data.map(lambda x: ( (x['author'], x['subreddit']), 1 )).reduceByKey(lambda a,b: a+b)
    author_subreddit_submission = author_subreddit_submission.map(lambda x: {'author': x[0][0], 'subreddit': x[0][1], 'submissions': x[1]})

    ### author-comment frequencies
    coms_data = sc.pickleFile(coms_path) # corpora-reddit/corpus-comments
    coms_data = coms_data.filter(lambda x: True if all (k in x for k in ('author', 'subreddit')) else False)
    coms_data = coms_data.filter(lambda x: x['author'] in authors_from_pairs)

    author_subreddit_comment = coms_data.map(lambda x: ( (x['author'], x['subreddit']), 1 )).reduceByKey(lambda a,b: a+b)
    author_subreddit_comment = author_subreddit_comment.map(lambda x: {'author': x[0][0], 'subreddit': x[0][1], 'comments': x[1]})

    data1 = author_subreddit_submission.map(extend).map(lambda x: ((x['author'], x['subreddit']), x) )
    data2 = author_subreddit_comment.map(extend).map(lambda x: ((x['author'], x['subreddit']), x) )

    united = data1.union(data2).reduceByKey(reduce_subreddit)
    print('united:', united.count())

    author_subreddit_union = united.map(lambda x: {'author': x[1]['author'], 'submissions': x[1]['submissions'], 'subreddit': x[1]['subreddit'],\
        'comments': x[1]['comments']})

    ### Filter user profile subreddits
    print('author_subreddit_union:', author_subreddit_union.count())
    author_subreddit_union = author_subreddit_union.filter(lambda x: not x['subreddit'].startswith('u_') and x['subreddit'] != '')
    print('author_subreddit_union (user profiles filtered):', author_subreddit_union.count())

    ### Filter bots and group by author
    author_subreddit_botlist = author_subreddit_union.filter(lambda x: x['author'] not in botlist.value)
    author_subreddit_grouped = author_subreddit_botlist.map(lambda x: (x['author'], x)).groupByKey().mapValues(list)
    print('author_subreddit_botlist:', author_subreddit_botlist.count())
    print('author_subreddit_grouped:', author_subreddit_grouped.count())

    author_subreddit = author_subreddit_grouped.map(lambda x: {"author": x[0], "subreddits": x[1]})

    ### only authors from 'changemyview'
    author_subreddit_cmv = author_subreddit.filter(lambda x: filter_cmv(x))

    author_subreddit_json = author_subreddit_cmv.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8') )
    
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_subreddit_json.saveAsTextFile(output_path)

def filter_cmv(x):
    for s in x['subreddits']:
        if s['subreddit'] == 'changemyview':
            return True
    return False

def extend(x):
    if 'submissions' not in x and 'delta_submissions' not in x:
        x['submissions'] = 0
        x['delta_submissions'] = 0
    if 'comments' not in x:
        x['comments'] = 0

    return x

def reduce_subreddit(a, b):
    joined = {}
    joined['author'] = a['author']
    joined['subreddit'] = a['subreddit']
    joined['submissions'] = a['submissions'] + b['submissions']
    joined['comments'] = a['comments'] + b['comments']

    return joined

def jsonloads(l):
    try:
        return json.loads(l)
    except:
        return ""
