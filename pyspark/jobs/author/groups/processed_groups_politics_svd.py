"""filter subreddits in News and Politics category"""

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
import scipy
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

    def chunks(DATA, SIZE):
        SIZE = min(samples,SIZE)
        for i in range(0, int(samples/SIZE)):
            to = samples if samples-(i+1)*SIZE < SIZE else (i+1)*SIZE
            yield DATA[i*SIZE:to]

    dv = DictVectorizer()
    dv.fit(input_dict.values())
    for k,v in input_dict.items():
        input_dict[k] = dv.transform(v)

    author_index = list(input_dict.keys())
    data = list(input_dict.values())
    samples = len(author_index)
    features = len(dv.get_feature_names())

    X = scipy.sparse.vstack(data)

    n_components = 3
    if n_components >= features or n_components >= samples:
        n_components = min(features,samples)

    Xtransformed = None
    sklearn_pca = IncrementalPCA(n_components=n_components)
    for chunk in chunks(X, 1000):
        c = chunk.toarray()
        sklearn_pca.partial_fit(c)
    for chunk in chunks(X, 1000):
        c = chunk.toarray()
        Xchunk = sklearn_pca.transform(c)
        if Xtransformed is None:
            Xtransformed = Xchunk
        else:
            Xtransformed = np.vstack((Xtransformed, Xchunk))


    p = pd.DataFrame(Xtransformed, columns=['%i' % i for i in range(n_components)], index=author_index)

    output_list = []
    for k,v in zip(p.index, p.values):
        output_list.append((k,v))
    return (subreddit, output_list)

def get_index():
    file_path = '/home/username/data/subreddits-by-category-final.json'
    with open(file_path) as f:
        data = json.load(f)
    categories = {}
    for c in data['category']:
        categories[c['name']] = []
        if c.get('subcategory', None):
            for subc in c['subcategory']:
                categories[c['name']].append(subc['name'])
                if subc.get('subcategory', None):
                    for subc2 in subc['subcategory']:
                        categories[c['name']].append(subc2['name'])
    return categories

cat_index = get_index()
cats_politics = cat_index['News and Politics']

ae_flat_cat = sc.pickleFile('/user/username/data/output/_jobs/author_entities_cats')
ae_flat_cat = ae_flat_cat.filter(lambda x: x[5][0] in cats_politics)

medians = ae_flat_cat.map(lambda x: ((x[5][0],x[0],x[1]),x[3]) ).groupByKey().mapValues(list).map(lambda x: x[0]+(float(np.median(x[1])),) )
### filter out AskReddit
ae_grouped_by_subreddit = medians.groupBy(lambda x: (x[0])).mapValues(list)

processed_groups = ae_grouped_by_subreddit.map(lambda x: process_group(x)).filter(lambda x: len(x)>0)
processed_groups.saveAsPickleFile('/user/username/data/output/_jobs/processed_groups_politics_pca')


