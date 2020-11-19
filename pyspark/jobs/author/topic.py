"""author-topic (wikipedia main topics) vectors from author-entity"""

from pyspark import SparkContext, SparkConf, SparkFiles, AccumulatorParam, Accumulator
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
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))
    local_output_root = kwargs.get('local-output-root', None)
    page_topics_pickle = kwargs['page-topic-pickle']

    page_topics = pickle.load(open(page_topics_pickle, 'rb'))
    _pt_dict = {}
    for p in page_topics:
        _pt_dict.update({p[0]: p[1]})
    pt_dict = sc.broadcast(_pt_dict)

    pages_other = sc.accumulator([], ListParam())

    author_entity = sc.pickleFile(input_path)

    author_topic = author_entity.map(lambda x: entities_to_topics(x, pt_dict, pages_other) )

    print('pages_other:')
    pp.pprint(pages_other.value)

    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    author_topic.saveAsPickleFile(output_path)


class ListParam(AccumulatorParam):
    def zero(self, v):
        return []
    def addInPlace(self, acc1, acc2):
        acc1.extend(acc2)
        return acc1

def entities_to_topics(x, pt_dict, pages_other):
    res = {}
    author = x[0]
    entities = x[1]
    topics_freq = {} # instead of counts use topic frequency from page-topics
    topics = {}
    topics_pages = {}
    topics_sent = {} # topic-sentiment
    for e in entities:
        pid = e[0]
        count = e[1][1]

        ptopics = pt_dict.value.get(pid, False)
        if ptopics is False:
            pages_other += [pid]
            ptopics = {'Other': 0}

        for pt, pt_f in ptopics.items():
            if pt not in topics_pages:
                topics_pages[pt] = {}
            topics_pages[pt][pid] = count

            if pt not in topics:
                topics[pt] = count
            else:
                topics[pt] += count

            if pt not in topics_freq:
                topics_freq[pt] = pt_f
            else:
                topics_freq[pt] += pt_f

            if pt not in topics_sent:
                topics_sent[pt] = []
            topics_sent[pt].append(e[1][0][4])

    res['author'] = author
    res['topics'] = topics
    res['topics_pages'] = topics_pages
    res['topics_freq'] = topics_freq
    res['topics_sent'] = topics_sent

    return res
