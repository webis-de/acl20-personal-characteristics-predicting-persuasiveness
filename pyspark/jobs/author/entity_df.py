"""calculate DF for entities"""

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

    data = sc.pickleFile('/user/username/data/output/_jobs/author_entity_freq')

    flattened = data.flatMap(lambda x: [e[0][2] for e in x[1]] )
    reduced = flattened.map(lambda x: (x, 1) ).reduceByKey(lambda a,b: a+b)
    mapped = reduced.map(lambda x: {"entity_id": x[0], "df": x[1]})

    print('Saving to {0}...'.format(local_output_root+'/entity_df.pickle'))
    collected = mapped.collect()
    _df_dict = {}
    for s in collected:
        _df_dict.update({s['entity_id']: s['df']})
    pickle.dump(_df_dict, open(local_output_root+'/entity_df.pickle','wb'))
