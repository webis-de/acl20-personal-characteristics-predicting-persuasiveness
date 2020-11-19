from scipy.sparse import csr_matrix
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn import preprocessing
import json
import pprint
import pickle
import pandas as pd
import numpy as np
pp = pprint.PrettyPrinter(width=100)

def process_group(grouped):
    subreddit = grouped[0]
    d = grouped[1]
    triples = list(map(lambda x: (x[0],x[1],x[3]), d))
    data = {}
    for x in d:
        if x[0] not in data:
            data[x[0]] = {}
        data[x[0]].update({x[1]: x[3]})
    X = pd.DataFrame.from_dict(data, orient='index')
    ### todo: how to use mean sentiment score dict? when???
    # mean_score = X.groupby('entity_id')['score'].mean().to_dict()
    # entities = defaultdict(lambda: len(entities))
    # data['entityid'] = data['entity_id'].map(entities.__getitem__)
    # X = data.pivot_table(index='author',columns='entity_id', values='score')
    n_components=3
    if n_components > min(X.shape[0], X.shape[1]):
        return tuple()
    X = X.fillna(0) #X.mean()
    # pca = PCA(n_components)
    # p = pd.DataFrame(pca.fit_transform(X), columns=['%i' % i for i in range(n_components)], index=X.index)
    # dtype = dict(names=['id','data'],formats=['i','f8'])
    # return list((k,{subreddit: np.array(list(v.items()), dtype=dtype) }) for k,v in p.to_dict(orient='index').items())
    # return p
    return X.shape

ae_flat = sc.pickleFile('/user/username/data/output/_jobs/author_entities_flat_distinct')
ae_grouped_by_subreddit = ae_flat.groupBy(lambda x: (x[4])).mapValues(list)

subreddit_category = pickle.load(open('/home/username/data/output/_jobs/subreddit_category.pickle', 'rb'))
subreddit_topcategory = pickle.load(open('/home/username/data/output/_jobs/subreddit_topcategory.pickle','rb'))

# test_df = ae_grouped_by_subreddit.take(20)
processed_groups = ae_grouped_by_subreddit.map(lambda x: process_group(x))
processed_groups.take(1)
processed_groups.filter(lambda x: len(x) == 0).count()
processed_groups.saveAsPickleFile('/user/username/data/output/_jobs/processed_groups')
