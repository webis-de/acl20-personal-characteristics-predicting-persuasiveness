#!/usr/bin/python
import argparse
import importlib
import time
import os
import sys
import pyspark
import pprint
import configparser


sys.path.insert(0, 'jobs.zip')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a PySpark job')
    parser.add_argument('--job', type=str, required=True, dest='job_name', help="The name of the job module you want to run. (ex: poc will run job on jobs.poc package)")
    parser.add_argument('--job-args', nargs='*', help="Extra arguments to send to the PySpark job (example: --job-args template=manual-email1 foo=bar")

    args = parser.parse_args()
    print("Called with arguments: %s" % args)

    job_args = dict()
    if args.job_args:
        job_args_tuples = [arg_str.split('=') for arg_str in args.job_args]
        # print('job_args_tuples: %s' % job_args_tuples)
        job_args = {a[0]: a[1] for a in job_args_tuples}

    config = configparser.ConfigParser()
    config.read('config.ini')
    job_args.update(dict(config.items('default')))
    job_args['job-name'] = args.job_name

    pp = pprint.PrettyPrinter(width=100)
    job_args['pp'] = pp

    environment = {
        'PYSPARK_JOB_ARGS': ' '.join(args.job_args) if args.job_args else ''
    }
    print('\nRunning job %s...\nEnvironment: %s' % (args.job_name, environment))

    os.environ.update(environment)
    sc = pyspark.SparkContext(appName=args.job_name, environment=environment)

    sc.addPyFile('dist/jobs.zip')
    # sys.path.insert(0, SparkFiles.get("dist/jobs.zip"))
    print('\nsparkConf: %s' % sc.getConf().getAll())

    job_module = importlib.import_module('jobs.%s' % args.job_name)

    start = time.time()
    job_module.analyze(sc, **job_args)
    end = time.time()

    print("\nExecution of job %s took %s seconds" % (args.job_name, end-start))
