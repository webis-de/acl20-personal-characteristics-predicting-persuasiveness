"""get list of CMV authors"""

from pyspark import SparkContext, SparkConf, SparkFiles
import json
import pickle
import csv
import time
from jobs.shared import utils

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    timestamp = int(time.time())
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']
    input_path = kwargs['input-path'] # /user/username/data/output/_jobs/author_subreddit

    collected = sc.textFile(input_path).map(lambda x: json.loads(x)).map(lambda x: x['author']).collect()

    pickle.dump(collected, open(local_output_root+'/data/output/_jobs/cmv_authors.pickle', 'wb'))
