"""DF for subreddits"""

from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
import pprint
import pickle

pp = pprint.PrettyPrinter(width=120)

def analyze(sc, **kwargs):
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs.get('local-output-root', None)
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))

    file = sc.textFile(input_path) # /user/username/data/output/author_subreddits/latest
    data = file.map(lambda l : json.loads(l) )

    flattened = data.flatMap(lambda x: [y for y in x['subreddits']] )
    reduced = flattened.map(lambda x: (x['subreddit'], 1) ).reduceByKey(lambda a,b: a+b)
    mapped = reduced.map(lambda x: {"subreddit": x[0], "df": x[1]})

    subreddit_df_json = mapped.map(lambda l: json.dumps(l, ensure_ascii=False).encode('utf8') )
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    subreddit_df_json.saveAsTextFile(output_path)

    print('Saving to {0}...'.format(local_output_root+'/subreddit_df.pickle'))
    collected = mapped.collect()
    _df_dict = {}
    for s in collected:
        _df_dict.update({s['subreddit']: s['df']})
    pickle.dump(_df_dict, open(local_output_root+'/subreddit_df.pickle','wb'))
