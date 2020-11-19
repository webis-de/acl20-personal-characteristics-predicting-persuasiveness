from pyspark import SparkConf, SparkContext
from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing
import math
import json
import pprint
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
pp = pprint.PrettyPrinter(width=100)

conf = SparkConf().setAppName("process_groups")
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

    if len(input_dict.items()) < 2:
        return tuple()

    dv = DictVectorizer()
    dv.fit(input_dict.values())

    for k,v in input_dict.items():
        input_dict[k] = dv.transform(v)

    author_index = list(input_dict.keys())
    data = list(input_dict.values())
    samples = len(author_index)
    features = len(dv.get_feature_names())

    if features < 2:
        return tuple()

    n_components=min(int(round(math.sqrt(features))),samples-1 if samples > 1 else samples)

    X = vstack(data)
    pca = TruncatedSVD(n_components)
    Xtransformed = pca.fit_transform(X)

    p = pd.DataFrame(Xtransformed, columns=['%i' % i for i in range(n_components)], index=author_index)

    output_list = []
    for k,v in zip(p.index, p.values):
        output_list.append((k,v))
    return (subreddit, output_list)

ae_flat_cat = sc.pickleFile('/user/username/data/output/_jobs/author_entities_cats')

# map as ((subcat, author, entity_id), score) and compute median
medians = ae_flat_cat.map(lambda x: ((x[5][1],x[0],x[1]),x[3]) ).groupByKey().mapValues(list).map(lambda x: x[0]+(float(np.median(x[1])),) )

#group by cat
ae_grouped_by_subreddit = medians.groupBy(lambda x: (x[0])).mapValues(list)

processed_groups = ae_grouped_by_subreddit.map(lambda x: process_group(x)).filter(lambda x: len(x)>0)
processed_groups.saveAsPickleFile('/user/username/data/output/_jobs/processed_groups_topcats_svd')


