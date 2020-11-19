"""author-sentences from subreddits' submissions/comments"""

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

pp = pprint.PrettyPrinter(width=100)

def analyze(sc, **kwargs):
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    subs_path = kwargs['subs-path']
    coms_path = kwargs['coms-path']
    botlist_csv = kwargs['botlist-csv']
    subs_path = kwargs['subs-path']
    coms_path = kwargs['coms-path']

    sc.addFile("hdfs://hadoop:8020/user/username/nltk_data", recursive = True)

    with open(botlist_csv, 'r') as f:
        reader = csv.reader(f)
        _botlist = list(map(lambda x: x[0], list(reader)))
    botlist = sc.broadcast(_botlist)

    subs_file = sc.textFile(subs_path)
    subs_data = subs_file.map(lambda l : jsonloads(l) ).filter(lambda l : l is not "" and 'author' in l and 'selftext' in l)

    coms_file = sc.textFile(coms_path)
    coms_data = coms_file.map(lambda l : jsonloads(l) ).filter(lambda l : l is not ""  and 'author' in l and 'body' in l)

    print("\n#Filtering sub/com...")
    subs_data = subs_data.filter(lambda x: x['author'] not in botlist.value and len(x["selftext"]) > 0 and x["selftext"] != "[removed]")
    coms_data = coms_data.filter(lambda x: x['author'] not in botlist.value and len(x["body"]) > 0 and x["body"] != "[removed]")

    cmv_authors = pickle.load(open('/home/username/data/output/cmv_authors.pickle', 'rb'))
    cmv_authors_b = sc.broadcast(cmv_authors)
    subs_data = subs_data.filter(lambda x: x['author'] in cmv_authors_b.value)
    coms_data = subs_data.filter(lambda x: x['author'] in cmv_authors_b.value)

    print("\n#Running nltk...")
    subs_sentences = subs_data.map(lambda x: get_sentences(x, 'selftext')).map(lambda x: (x['author'], x['sentences']) )
    coms_sentences = coms_data.map(lambda x: get_sentences(x, 'body')).map(lambda x: (x['author'], x['sentences']) )
    print(subs_sentences.take(1))

    print("\n#Union, flatten, reduce...")
    united = sc.union([subs_sentences, coms_sentences])
    print(united.take(1))

    sentences_reduced = united.reduceByKey(lambda a,b: a+b)
    print(sentences_reduced.take(1))

    print("\n#Saving results...")
    output_json = sentences_reduced.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    output_json.saveAsTextFile(output_path)


def get_sentences(s, prop):
    import nltk
    nltk.data.path.append(SparkFiles.get("nltk_data"))

    from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
    from nltk.corpus import stopwords

    try:
        text = s[prop]
    except Exception as e:
        print(s)
        s['sentences'] = []
        return s

    if "_____" in text:
        text = text.split("_____")[0]
    if "Hello, users of CMV!" in text:
        text = text.split("Hello, users of CMV!")[0]

    sentences = sent_tokenize(text)
    sentences_processed = []
    for se in sentences:
        stops = stopwords.words('english') + list(string.punctuation)
        wo_stops = " ".join([word for word in re.findall('\w+', se) if word not in stops])
        sentences_processed.append(wo_stops)
    s['sentences'] = sentences_processed

    return s

def jsonloads(l):
    try:
        return json.loads(l)
    except:
        return ""

