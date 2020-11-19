'''
Historical author features:
- total # words from posts across all subreddits
- # stopwords
'''

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

    authors_from_pairs = set(pickle.load(open('/home/username/data/output/_jobs/authors_from_pairs.pickle','rb')))
    print('# authors_from_pairs:', len(authors_from_pairs))

    subs_pairauthors2 = sc.pickleFile('/user/username/data/output/_jobs/cleaned_cmv_authors2/subs')\
        .filter(lambda x: x['author'] in authors_from_pairs)
    coms_pairauthors2 = sc.pickleFile('/user/username/data/output/_jobs/cleaned_cmv_authors2/coms')\
        .filter(lambda x: x['author'] in authors_from_pairs)

    sc.addFile("hdfs://hadoop:8020/user/username/nltk_data", recursive = True)

    print("\n#Running nltk, semanticizer...")
    subs_words = subs_pairauthors2.map(lambda x: process_post(x, 'selftext'))
    coms_words = coms_pairauthors2.map(lambda x: process_post(x, 'body'))

    print("\n#Union, flatten, reduce...")
    united = sc.union([subs_words, coms_words])
    num_words_reduced = united.reduceByKey(lambda a,b: (a[0]+b[0],a[1]+b[1]))

    num_words_reduced.saveAsPickleFile('/user/username/data/output/_jobs/author_num_words')

    author_num_words = num_words_reduced.collect()

    _dict = {}
    for x in author_num_words:
        _dict[x[0]] = (x[1][0],x[1][1])
    pickle.dump(_dict, open('/home/username/data/output/_jobs/features/author_history_words_dict.pickle','wb'))


def process_post(s, prop):
    import nltk, string
    nltk.data.path.append(SparkFiles.get("nltk_data"))
    from nltk.corpus import stopwords

    text = s[prop]

    if prop == 'selftext':
        if "_____" in text and "Hello, users of CMV!" in text and not text.startswith("_____"):
            text = text.split("_____")[0]
        if "This is a footnote from the CMV moderators." in s['selftext']:
            text = text.split("This is a footnote from the CMV moderators.")[0]

    post_words = nltk.word_tokenize(text)
    num_words = len(post_words)

    stops = stopwords.words('english') + list(string.punctuation)
    num_stops = len([word for word in post_words if word not in stops])

    return (s['author'], (num_words, num_stops))



