"""
- run wikifier on submissions/comments from all subreddits
- return groupedByAuthor: _jobs/author_entity/latest (author, [(25536, (('Republic', 'Q7270', 25536, 'republic', 0.5106, 'worldnews'), 2)])
"""

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
    botlist_csv = kwargs['botlist-csv'] # '/home/username/data/botlist.csv'
    stopwords_csv = kwargs['stopwords-csv'] # /home/username/data/ranksnl_stopwords.csv
    sem_model_path = kwargs['sem-model-path'] # 'hdfs://hadoop:8020/user/username/data/enwiki_pages_n3_1.model'
    sem_path = kwargs['sem-path'] # '/home/username/tools/semanticizest/semanticizest3.zip'
    subs_path = kwargs['subs-path']
    coms_path = kwargs['coms-path']

    sc.addFile(sem_model_path)
    sc.addPyFile(sem_path)

    sys.path.insert(0, SparkFiles.get('libs.zip'))
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

    import nltk
    nltk.data.path.append(SparkFiles.get("nltk_data"))

    from nltk import ne_chunk, pos_tag, word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.corpus import wordnet

    from nltk.stem import WordNetLemmatizer
    lemmatizer = sc.broadcast(WordNetLemmatizer())

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    analyser = sc.broadcast(SentimentIntensityAnalyzer())


    subs_data = sc.pickleFile(subs_path)
    coms_data = sc.pickleFile(coms_path)

    print("\n#Running nltk, semanticizer...")
    subs_entities = subs_data.map(lambda x: process_sentences(x, 'selftext', sem, lemmatizer, analyser))
    coms_entities = coms_data.map(lambda x: process_sentences(x, 'body', sem, lemmatizer, analyser))

    print("\n#Union, flatten, reduce...")
    united = sc.union([subs_entities, coms_entities])
    entities_reduced = united.reduceByKey(lambda a,b: (a+b))

    ### remove duplicates
    author_entity_cleaned = entities_reduced.map(lambda x: (x[0],list(set(x[1]))) )

    ### calculate entity frequencies
    ### TODO: used for freq-entity vector jobs, run separately from this job
    entities_flat = author_entity_cleaned.flatMap(lambda x: [ ((x[0], e[2]), (e, 1)) for e in x[1]] )
    entities_reduced = entities_flat.reduceByKey(lambda a,b: (a[0], a[1]+b[1]))
    entities_grouped_by_author = entities_reduced.map(lambda x: (x[0][0],x[1]) ).groupByKey().mapValues(list)
    entities_grouped_by_author.saveAsPickleFile('/user/username/data/output/_jobs/author_entity_freq')

    ### add category and top-category to entities
    ### TODO: run as a separate job
    subreddit_category = pickle.load(open('/home/username/data/output/_jobs/subreddit_category_index.pickle','rb'))
    subreddit_topcategory = pickle.load(open('/home/username/data/output/_jobs/subreddit_topcategory.pickle','rb'))
    entities_flat = author_entity_cleaned.flatMap(lambda x: [ (e[0], e[1], e[2], e[5]) for e in x[1]] ).distinct()
    entities_categories = entities_flat.map(lambda x: x + (subreddit_category.get(x[3],'Other'),
                                                           subreddit_topcategory.get(x[3], 'Other')) )
    entities_categories.saveAsPickleFile('/user/username/data/output/_jobs/author_entity_categories')

    ### create two dictionaries (top)category->list of entities
    entities_categories = sc.pickleFile('/user/username/data/output/_jobs/author_entity_categories')
    _entities_categories = entities_categories.map(lambda x: (x[2],x[4],x[5])).collect()
    category_entities = {}
    topcategory_entities = {}
    for x in _entities_categories:
        e = x[0]
        cat = x[1]
        topcat = x[2]
        if cat not in category_entities:
            category_entities[cat] = {}
        category_entities[cat][e] = 0
        if topcat not in topcategory_entities:
            topcategory_entities[topcat] = {}
        topcategory_entities[topcat][e] = 0
    pickle.dump(category_entities, open('/home/username/data/output/_jobs/category_entities.pickle','wb'))
    pickle.dump(topcategory_entities, open('/home/username/data/output/_jobs/topcategory_entities.pickle','wb'))

    ### create dictionary entity-categories
    ### TODO: run as a separate job
    entity_categories_dict = {}
    for x in _entities_categories:
        e = x[0]
        cat = x[1]
        topcat = x[2]
        if e not in entity_categories_dict:
            entity_categories_dict[e] = {}
        entity_categories_dict[e]['cat'] = cat
        entity_categories_dict[e]['topcat'] = topcat
    pickle.dump(entity_categories_dict, open('/home/username/data/output/_jobs/entity_categories_dict.pickle','wb'))


    print("\n#Saving results...")
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_entity_cleaned.saveAsPickleFile(output_path)


def semanticize(sentence, sem, sent_score, subreddit, sub_id):
    all_entities = []
    entities = set(sem.value.all_candidates(sentence))
    entities = sorted( filter(lambda e : e[3] > 0.03, entities) , key=lambda x: -x[3])
    entities = map(lambda x: (x[2], x[4], x[6], x[7], sent_score, subreddit, sub_id), entities)
    all_entities.extend(entities)
    return all_entities

def process_sentences(s, prop, sem, lemmatizer, analyser):
    import nltk
    nltk.data.path.append(SparkFiles.get("nltk_data"))
    from nltk import sent_tokenize
    from nltk.corpus import stopwords
    from nltk.corpus import wordnet 

    def nltk2wn_tag(nltk_tag):
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

    def lemmatize_sentence(sentence, lemmatizer):
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
        wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
        res_words = []
        for word, tag in wn_tagged:
            if tag is None:            
                res_words.append(word)
            else:
                res_words.append(lemmatizer.value.lemmatize(word, tag))

        return " ".join(res_words)

    if 'subreddit' not in s:
        print('subreddit not in s:', s)
        return (s['author'], [])
    try:
        text = s[prop]
    except Exception as e:
        print(s)
        return (s['author'], [])

    ### remove footer from submission
    if prop == 'selftext':
        sub_id = s['id']
        if "_____" in text and "Hello, users of CMV!" in text and not text.startswith("_____"):
            text = text.split("_____")[0]
        if "This is a footnote from the CMV moderators." in s['selftext']:
            text = text.split("This is a footnote from the CMV moderators.")[0]
    else:
        sub_id = s['link_id']

    sentences = sent_tokenize(text)
    sentences_processed = []
    all_entities = []
    for se in sentences:
        sent_score = analyser.value.polarity_scores(se)['compound']
        lemm = lemmatize_sentence(se.lower(), lemmatizer)
        subreddit = s['subreddit']

        stops = stopwords.words('english') + list(string.punctuation)
        wo_stops = " ".join([word for word in re.findall('\w+', lemm) if word not in stops])

        entities = semanticize(wo_stops, sem, sent_score, subreddit, sub_id)
        all_entities.extend(entities)

    return (s['author'], all_entities)


def jsonloads(l):
    try:
        return json.loads(l)
    except:
        return ''

