from pyspark import SparkConf, SparkContext
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn import preprocessing
import math
import json
import pprint
import pickle
import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction import DictVectorizer
pp = pprint.PrettyPrinter(width=100)

conf = SparkConf().setAppName("process_groups") # setMaster("local").
sc = SparkContext(conf = conf)

def process_group(grouped):
    subreddit = grouped[0]
    d = grouped[1]
    input_dict = {}
    for x in d:
        a_id = x[1]
        e_id = x[2]
        score = x[3]
        if a_id not in input_dict:
            input_dict[a_id] = {}
        input_dict[a_id].update({e_id: score})

    if len(input_dict.items()) < 1:
        return tuple()

    dv = DictVectorizer()
    dv.fit(input_dict.values())

    for k,v in input_dict.items():
        input_dict[k] = dv.transform(v)

    author_index = list(input_dict.keys())
    X = list(input_dict.values())
    samples = len(author_index)
    features = len(dv.get_feature_names())

    def chunks(data, SIZE):
        SIZE = min(samples,SIZE)
        for i in range(0, int(samples/SIZE)):
            to = samples if samples-(i+1)*SIZE < SIZE else (i+1)*SIZE
            yield data[i*SIZE:to]

    if features < 2:
        return tuple()
    n_components=min(int(round(math.sqrt(features))),samples-1 if samples > 1 else samples)

    Xtransformed = None
    sklearn_pca = IncrementalPCA(n_components=n_components)
    for chunk in chunks(X, 1000):
        c = scipy.sparse.vstack(chunk).toarray()
        sklearn_pca.partial_fit(c)
    for chunk in chunks(X, 1000):
        c = scipy.sparse.vstack(chunk).toarray()
        Xchunk = sklearn_pca.transform(c)
        if Xtransformed is None:
            Xtransformed = Xchunk
        else:
            Xtransformed = np.vstack((Xtransformed, Xchunk))

    p = pd.DataFrame(Xtransformed, columns=['%i' % i for i in range(n_components)], index=author_index)
    dtype = dict(names=['id','data'],formats=['i','f8'])
    return list((k,{subreddit: np.array(list(v.items()), dtype=dtype) }) for k,v in p.to_dict(orient='index').items())

ae_flat = sc.pickleFile('/user/username/data/output/_jobs/author_entities_flat_distinct')
# ae_flat_cat:  ('CommitteeOfOne', 40147, 'divorce', 0.0, 'AskReddit', ('Discussion', 'General'))
# map as ((subreddit, author, e_id))
medians = ae_flat.map(lambda x: ((x[4],x[0],x[1]),x[3]) ).groupByKey().mapValues(list).map(lambda x: x[0]+(float(np.median(x[1])),) )
### filter out AskReddit
# ae_flat = ae_flat.filter(lambda x: x[4] != 'AskReddit')
ae_grouped_by_subreddit = medians.groupBy(lambda x: (x[0])).mapValues(list)

processed_groups = ae_grouped_by_subreddit.map(lambda x: process_group(x)).filter(lambda x: len(x)>0)
processed_groups.saveAsPickleFile('/user/username/data/output/_jobs/processed_groups_partial')


