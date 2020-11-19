"""split each pairs_wa into two separate samples: positive and negative"""

from pyspark import SparkContext, SparkConf, SparkFiles
import json
import pprint
import time
import pickle
from jobs.shared import utils

pp = pprint.PrettyPrinter(width=100)

def analyze(sc, **kwargs):
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))

    file = sc.textFile(input_path)
    pairs = file.map(lambda l: json.loads(l))

    split = pairs.flatMap(lambda x: split(x) )

    pickle.dump(split.collect(), open('/home/username/data/output/_jobs/pairs_winargs_split.pickle','wb'))

def split(p):
    pairs_split = []
    pairs_split.append((p[1],p[2],True))
    pairs_split.append((p[1],p[3],False))
    return pairs_split


