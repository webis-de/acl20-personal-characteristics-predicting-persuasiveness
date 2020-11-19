"""run wikifier on CMV submissions/comments"""

from pyspark import SparkContext, SparkConf, SparkFiles
import sys
import json
import pprint
import csv
import time
import string
import re
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
    stopwords_csv = kwargs['stopwords-csv'] # ranksnl_stopwords.csv
    sem_model_path = kwargs['sem-model-path']
    sem_path = kwargs['sem-path']
    subs_path = kwargs['subs-path']
    coms_path = kwargs['coms-path']

    sc.addFile(sem_model_path)
    sc.addPyFile(sem_path)

    sc.addFile("hdfs://hadoop:8020/user/username/nltk_data", recursive = True)

    print("\n#sematicize():")
    sys.path.insert(0, SparkFiles.get(sem_path.split('/')[-1]))
    from semanticizest import Semanticizer
    print("\n#loading model...")
    _sem = Semanticizer(SparkFiles.get(sem_model_path.split('/')[-1]))
    print("\n#model loaded.")
    sem = sc.broadcast(_sem)

    with open(stopwords_csv, 'r') as f:
        reader = csv.reader(f)
        _stopwords = list(map(lambda x: x[0], list(reader)))
    stopwords = sc.broadcast(_stopwords)

    with open(botlist_csv, 'r') as f:
        reader = csv.reader(f)
        _botlist = list(map(lambda x: x[0], list(reader)))
    botlist = sc.broadcast(_botlist)


    subs_file = sc.textFile(subs_path)
    subs_data = subs_file.map(lambda l : json.loads(l) )

    coms_file = sc.textFile(coms_path)
    coms_data = coms_file.map(lambda l : json.loads(l) )

    print("\n#Filtering sub/com...")
    subs_data = subs_data.filter(lambda x: x['author'] not in botlist.value and len(x["selftext"]) > 0 and x["selftext"] != "[removed]")
    coms_data = coms_data.filter(lambda x: x['author'] not in botlist.value and len(x["body"]) > 0 and x["body"] != "[removed]")

    print("\n#Running nltk...")
    subs_sentences = subs_data.map(lambda x: get_sentences(x, 'selftext'))
    coms_sentences = coms_data.map(lambda x: get_sentences(x, 'body'))

    print("\n#Running wikifier...")
    subs_entities = subs_sentences.map(lambda x: semanticize(x, sem))
    coms_entities = coms_sentences.map(lambda x: semanticize(x, sem))

    print("\n#Union, flatten, reduce...")
    united = sc.union([subs_entities, coms_entities])

    entities_flat = united.flatMap(lambda x: [ ((x['author'], e), 1) for e in x['entities']] )

    print(entities_flat.take(1))
    # [(('Toby-OrNotToby', ('HMV', 0.2189578713968958, 'Q10854572', 1144829, 'HMV')), 1)]
    entities_reduced = entities_flat.reduceByKey(lambda a,b: a+b)
    print(entities_reduced.take(1))
    # [(('disinterestedMarmot', ('Solutions of the Einstein field equations', 5.930494603249911e-05, 'Q4394061', 2001621, 'solution')), 9)]
    entities_grouped_by_author = entities_reduced.map(lambda x: (x[0][0], (x[0][1], x[1]) ) ).groupByKey().mapValues(list)
    print(entities_grouped_by_author.take(1))


    print("\n#Saving results...")
    output_json = entities_grouped_by_author.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    output_json.saveAsTextFile(output_path)


def semanticize(s, sem):
    print("\n#process sentences:")
    all_entities = []
    for i, se in enumerate(s['sentences']):
        entities = sem.value.all_candidates(se)
        entities = sorted( filter(lambda e : e[3] > 0.03, entities) , key=lambda x: -x[3])
        entities = map(lambda x: (x[2], x[4], x[6], x[7]), entities)
        all_entities.extend(entities)
        print("\n #"+str(i), end='')
    s['entities'] = all_entities
    return {'author': s['author'], 'entities': s['entities']}

def get_sentences(s, prop):
    import nltk
    nltk.data.path.append(SparkFiles.get("nltk_data"))

    from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
    from nltk.corpus import stopwords

    text = s[prop]

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

    entities = []
    tokenized = word_tokenize(text)
    chunks = ne_chunk(pos_tag(tokenized))
    for c in chunks.subtrees():
        if c.label() in ('PERSON', 'GPE'):
            entities.append(" ".join(w for w, t in c))
    s['ne'] = entities

    return s

def entitiesToList(x):
    return [(x['author'], e) for e in x['entities']]

