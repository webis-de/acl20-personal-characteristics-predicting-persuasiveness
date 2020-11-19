"""calculate DF for categories"""

from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
import pprint
import pickle

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs.get('local-output-root', None)
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))

    data = sc.pickleFile(input_path) # /user/username/data/output/author_category/latest

    flattened = data.flatMap(lambda x: [k for k,v in x['categories'].items()] )
    print(flattened, flattened.count())
    reduced = flattened.map(lambda x: (x, 1) ).reduceByKey(lambda a,b: a+b)
    print(reduced, reduced.count())
    mapped = reduced.map(lambda x: {"category": x[0], "df": x[1]})
    print(mapped, mapped.count())

    data_json = mapped.map(lambda l: json.dumps(l, ensure_ascii=False).encode('utf8') )
    print(data_json, data_json.count())
    output_path = utils.hdfs_get_output_path(hdfs_root, job_name)
    data_json.saveAsTextFile(output_path)

    print('Saving to {0}...'.format(local_output_root+'/category_df.pickle'))
    collected = mapped.collect()
    _df_dict = {}
    for s in collected:
        _df_dict.update({s['category']: s['df']})
    pickle.dump(_df_dict, open(local_output_root+'/category_df.pickle','wb'))
