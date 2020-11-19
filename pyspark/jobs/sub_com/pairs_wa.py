'''
WinArgs:
1. Two types of tasks with different input datasets:
    - top-level comments
    - rooted-path-unit (all comments by top-level comment author in a thread)
2. Changes:
    - filter out first month (no DeltaBot was active)
    - consider delta comments only by OP
    - consider only "discussion trees" with > 10 challengers (top-level comments authors)
    - rooted-path-unit: for each thread take only comments by top-level author
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
    file = sc.textFile(input_path)
    threads = file.map(lambda l: json.loads(l))
    ### filter out empty threads
    pairs = threads.flatMap(lambda x: top_level(x, botlist)).filter(lambda x: x)

    pickle.dump(pairs.collect(), open('/home/username/data/output/_jobs/pairs_winargs.pickle','wb'))


    ### Save authors from sub_com_pairs, used later to filter raw corpus
    ### TODO: Run separately from this job.
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
    ### filter out pairs where sub['selftext'] or com['body'] empty OR author account was deleted
    top_coms = list(filter(lambda com: com['author'] != '[deleted]' and com['body'] not in ['', '[removed]'] 
        and com['author'] not in set(botlist.value), sub['comments']))
    if len(top_coms) < 10 or op == '[deleted]' or sub['selftext'] in ['', '[removed]', '[deleted]'] or len(sub['selftext'].split(' ')) < 5:
        return []

    ### remove footer from submission
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
        ### cut off all comments after delta
        cut = []
        for c in comments:
            cut.append(c)
            if c['delta']:
                delta_subtree = True
                delta_submission = True
                break
        subtrees.append({'delta': delta_subtree, 'author': root_author, 'comments': cut})

    if delta_submission:
        return get_similar_pairs(s['id'], s, subtrees)
    else:
        return []


def process_subtree(comment, sub, op, root_author, comments):
    if comment['author'] == root_author:
        delta = is_delta(comment, op)
        c = {k:v for k,v in comment.items() if k not in ['children']}
        c['delta'] = delta
        comments.append(c)

    for com in comment['children']:
        process_subtree(com, sub, op, root_author, comments)

# for each delta comment select most similar nodelta comment
def get_similar_pairs(s_id, s, subtrees):
    new_pairs = []
    delta_subtrees = [t for t in subtrees if t['delta']]
    nodelta_subtrees = [t for t in subtrees if not t['delta']]
    for d in delta_subtrees:
        max_jac = 0
        most_similar = None
        # body1 = get_full_body(d['comments'])
        body1 = d['comments'][0]['body']
        if len(word_list(body1)) < 10 and len(d['comments']) > 1:
            body1 = d['comments'][0]['body'] + ' ' + d['comments'][1]['body']
        for n in nodelta_subtrees:
            # body2 = get_full_body(n['comments'])
            body2 = n['comments'][0]['body']
            jac = jaccard(word_list(body1), word_list(body2))
            if jac >= max_jac:
                max_jac = jac
                most_similar = n
        new_pairs.append( (s_id, s, d, most_similar, max_jac) )

    return new_pairs

# def get_full_body(comments):
#     full_body = ""
#     for c in comments:
#         full_body += ' ' + c['body']
#     return full_body

def is_delta(com, op):
    for c in com['children']:
        # consider only deltas given by OP
        if c['author'] == op:
            for x in c['children']:
                if x['author'] == 'DeltaBot':
                    return True
    return False

def nltk2wn_tag(nltk_tag):
    from nltk.corpus import wordnet
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def word_list(text):
    import nltk
    nltk.data.path.append(SparkFiles.get("nltk_data"))
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    lemmatizer = WordNetLemmatizer()

    tokenized = nltk.word_tokenize(text.lower())
    nltk_tagged = nltk.pos_tag(tokenized)
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        clean = re.findall('\w+', word)
        if clean:
            word = clean[0]
        else:
            continue
        if tag is None:            
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    stops = set(stopwords_list + list(string.punctuation))
    return [word for word in res_words if word not in stops]

def jaccard(a, b):
    a = set(a)
    b = set(b)
    intersection = float(len(a.intersection(b)))
    return intersection / (len(a) + len(b) - intersection)

