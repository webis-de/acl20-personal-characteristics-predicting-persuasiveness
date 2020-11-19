from pyspark import SparkContext, SparkConf, SparkFiles
import json
import pprint
import csv
import string
import pickle
from jobs.shared import utils
import re
import nltk

'''
1. debate.org:
- length
- tfidf: unigram, bigram, trigram
- politeness cues
- showing evidence
- sentiment
- subjectivity
- swear words
- ...
- links
- numbers
- exclamation marks
- questions

2. winninga_arguments:
Word category–based features
- #(in-)/definite articles
- #positive/negative words
- #2nd person pronoun
- #links
- #1st person pronouns
- #1st person plural pronoun
- #quotations

Word score–based features
- arousal
- valence

Entire argument features
- word entropy
- #sentences
- #paragraphs

interplay features:
- number of common words
- reply fraction
- OP fraction
- Jaccard

'''

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']

    stopwords_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    sc.addFile("hdfs://hadoop:8020/user/username/nltk_data", recursive = True)

    pairs_rdd = sc.textFile('/user/username/data/output/_jobs/sub_com_pairs/latest').map(json.loads)
    pairs_filtered = pairs_rdd.filter(lambda x: x[0]['selftext'] not in ['', '[removed]'] and x[1]['body'] not in ['', '[removed]'])
    
    pairs_features = pairs_filtered.map(lambda x: process_pair(x, stopwords_list))

    collected = pairs_features.collect()
    
    pickle.dump(collected, open('/home/username/data/output/_jobs/pairs_features.pickle','wb'))


def process_pair(pair, stopwords_list):
    nltk.data.path.append(SparkFiles.get("nltk_data"))

    words1 = word_list(pair[0]['selftext'], stopwords_list)
    words2 = word_list(pair[1]['body'], stopwords_list)

    pair[0]['f_words'] = words(pair[0]['selftext'], stopwords_list)
    pair[1]['f_words'] = words(pair[1]['body'], stopwords_list)
    pair[0]['f_entropy'] = Entropy(words1)
    pair[1]['f_entropy'] = Entropy(words2)

    pair[0]['fi_common'] = set(words1).intersection(set(words2))
    pair[0]['fi_jaccard'] = jaccard(words1,words2)

    return pair

def words(text, stopwords_list):
    lemm = ' '.join(nltk.word_tokenize(text.lower()))
    stops = stopwords_list + list(string.punctuation)
    count = len([word for word in re.findall('\w+', lemm) if word not in stops])
    return count

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

def word_list(text, stopwords_list):
    nltk.data.path.append(SparkFiles.get("nltk_data"))
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
    stops = stopwords_list + list(string.punctuation)
    return [word for word in res_words if word not in stops]

def is_delta(x):
    for c in x[1]:
        if c[1]['delta']:
            return True
    return False

def jaccard(a, b):
    a = set(a)
    b = set(b)
    intersection = float(len(a.intersection(b)))
    return intersection / (len(a) + len(b) - intersection)

def shannon(boe):
    from math import log2
    total = sum(boe.values()) 
    return sum(freq / total * log2(total / freq) for freq in boe.values())

def Entropy(string,base = 2.0):
    import math
    dct = dict.fromkeys(list(string))
    # frequencies
    pkvec =  [float(string.count(c)) / len(string) for c in dct]
    # entropy
    H = -sum([pk  * math.log(pk) / math.log(base) for pk in pkvec ])
    return H
