import time
from subprocess import call
from subprocess import Popen, PIPE
import os
from functools import reduce
from itertools import groupby


def hdfs_get_output_path(hdfs_root, job_name, output_folder = None):
    folder = output_folder if output_folder else '/latest'
    timestamp = int(time.time())
    job_path = "_".join(job_name.split("."))
    output_path = hdfs_root + '/' + job_path + folder

    if output_folder is None:
        print('exec hdfs dfs -ls ' + output_path)
        p = Popen('exec hdfs dfs -ls ' + output_path, shell=True, stdout=PIPE, stderr=PIPE)
        # std_out, std_err = p.communicate()
        lines = p.stdout.readlines()
        if lines:
            print('exec hdfs dfs -mv ' + output_path + ' ' + output_path[:-6] + str(timestamp))
            p2 = Popen('exec hdfs dfs -mv ' + output_path + ' ' + output_path[:-6] + str(timestamp), shell=True, stdout=PIPE, stderr=PIPE)
            time.sleep(5)
            p2.kill()
        # time.sleep(5)
        p.kill()

    return output_path

    # os.environ["HADOOP_HOME"] = "/opt/hadoop/"
    # os.environ["ARROW_LIBHDFS_DIR"] = os.environ["HADOOP_HOME"] + "lib/native/libhdfs.so"
    # fs = pyarrow.hdfs.connect('hdfs://hadoop/', 8020, user='username')
    # lst = fs.ls('/user/username/data/output/_jobs/', detail=False)
    # print('files: ', lst)

def get_input_path(hdfs_root, input_job, input_path):
    if input_job:
        input_job_path = "_".join(input_job.split("."))
        input_path = hdfs_root + '/' + input_job_path + '/latest'
    elif input_path is None:
        # logging.error('Either input-job or input-path should be specified!')
        sys.exit("Either input-job or input-path must be specified!")

    return input_path

def reduceByKey(func, iterable):
    get_first = lambda p: p[0]
    get_second = lambda p: p[1]
    return map(
        lambda l: (l[0], reduce(func, map(get_second, l[1]))),
        groupby(sorted(iterable, key=get_first), get_first)
    )

