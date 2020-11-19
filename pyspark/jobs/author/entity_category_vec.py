"""vectorise author-entity dictionaries"""

from pyspark import SparkContext, SparkConf, SparkFiles
from jobs.shared import utils
import json
from functools import reduce
import pprint
import math
import csv
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import normalize
from functools import reduce
from itertools import groupby

def analyze(sc, **kwargs):
    pp = kwargs['pp']
    job_name = kwargs['job-name']
    hdfs_root = kwargs['hdfs-root']
    local_output_root = kwargs['local-output-root']
    input_path = utils.get_input_path(hdfs_root, kwargs.get('input-job', None), kwargs.get('input-path', None))

    '''TODO:
        - load category_entities and topcategory_entities dicts
        - author_entity: group list of entities into categories / topcategories
    '''

    _subreddit_category = pickle.load(open('/home/username/data/output/_jobs/subreddit_category_index.pickle','rb'))
    subreddit_category = sc.broadcast(_subreddit_category)
    _subreddit_topcategory = pickle.load(open('/home/username/data/output/_jobs/subreddit_topcategory.pickle','rb'))
    subreddit_topcategory = sc.broadcast(_subreddit_topcategory)

    category_entities = pickle.load(open('/home/username/data/output/_jobs/category_entities.pickle','rb'))
    topcategory_entities = pickle.load(open('/home/username/data/output/_jobs/topcategory_entities.pickle','rb'))

    _entity_df = pickle.load(open('/home/username/data/output/_jobs/entity_df.pickle', 'rb'))
    entity_df = sc.broadcast(_entity_df)

    # create vectorizer for each category out of entity_df dataset
    # _vectorizers_cats = {}
    # for k,v in category_entities.items():
    #     dv = DictVectorizer()
    #     dv.fit_transform(v)
    #     _vectorizers_cats[k] = dv
    # vectorizers_cats = sc.broadcast(_vectorizers_cats)
    # pickle.dump(_vectorizers_cats, open('/home/username/data/output/_jobs/vectorizers_cats.pickle','wb'))
    _vectorizers_cats = pickle.load(open('/home/username/data/output/_jobs/vectorizers_cats.pickle','rb'))
    vectorizers_cats = sc.broadcast(_vectorizers_cats)

    # _vectorizers_topcats = {}
    # for k,v in topcategory_entities.items():
    #     dv = DictVectorizer()
    #     dv.fit_transform(v)
    #     _vectorizers_topcats[k] = dv
    # vectorizers_topcats = sc.broadcast(_vectorizers_topcats)
    # pickle.dump(_vectorizers_topcats, open('/home/username/data/output/_jobs/vectorizers_topcats.pickle','wb'))
    _vectorizers_topcats = pickle.load(open('/home/username/data/output/_jobs/vectorizers_topcats.pickle','rb'))
    vectorizers_topcats = sc.broadcast(_vectorizers_topcats)


    # author_entities_categories = sc.pickleFile('/user/username/data/output/_jobs/author_entity_categories')
    data = sc.pickleFile('/user/username/data/output/_jobs/author_entity/latest')
    authors_total = data.count()

    ### calculating category_entities features vectors
    author_entity_features = data.map(lambda x: get_feature_vectors(x, authors_total, entity_df, vectorizers_cats,
        vectorizers_topcats, subreddit_category, subreddit_topcategory))


    output_path = utils.hdfs_get_output_path(hdfs_root, job_name) # output_folder='/tf_squared'
    author_entity_features.saveAsPickleFile(output_path)

def get_feature_vectors(author, authors_total, entity_df, vectorizers_cats,
        vectorizers_topcats, subreddit_category, subreddit_topcategory):
    author = entities_to_categories(author, subreddit_category, subreddit_topcategory)
    entities = author[1]

    ### map entities as ((id, cat), 1)
    ### reduceByKey and get counts
    ### groupby cat
    ### vectorize
    def group_map(entities_list, prop):
        res = map(lambda x: ((x[2], x[prop]), 1), entities_list)
        reduced = utils.reduceByKey(lambda a,b: a+b, res)
        mapped = list(map(lambda x: (x[0][0], x[0][1], x[1]), reduced))
        return mapped

    entities_grouped_by_cat = group_map(entities, 7)
    entities_grouped_by_topcat = group_map(entities, 8)

    def to_dict(entities_list):
        _dict = {}
        for e in entities_list:
            e_id = e[0]
            cat = e[1]
            count = e[2]
            if cat not in _dict:
                _dict[cat] = {}
            _dict[cat][e_id] = count
        return _dict

    dict_cat = to_dict(entities_grouped_by_cat)
    dict_topcat = to_dict(entities_grouped_by_topcat)

    def get_vecs(_dict, vectorizers):
        _vecs = {}
        for cat, entities in _dict.items():
            all_terms_in_doc = sum(list(entities.values()))
            tfidf = {}
            lognorm = {}
            raw = {}
            for k,v in entities.items():
                try:
                    count = v
                    idf = (math.log(authors_total / entity_df.value[k]))
                    tfidf[k] = (count / all_terms_in_doc) * idf
                    lognorm[k] = math.log(1 + count) * idf
                    raw[k] = count
                except KeyError:
                    print(k)        

            _vecs[cat] = {}
            _vecs[cat]['tfidf'] = vectorizers.value[cat].transform(tfidf)
            _vecs[cat]['lntfidf'] = vectorizers.value[cat].transform(lognorm)
            _vecs[cat]['raw'] = vectorizers.value[cat].transform(raw)
        return _vecs

    catent_vec = get_vecs(dict_cat, vectorizers_cats)
    topcatent_vec = get_vecs(dict_topcat, vectorizers_topcats)

    return {'author': author[0], 'catent_vec': catent_vec, 'topcatent_vec': topcatent_vec}

### Add categories properties for each entity
def entities_to_categories(author, subreddit_category, subreddit_topcategory):
    entities = []
    for e in author[1]:
        entities.append(e + (subreddit_category.value.get(e[5],'Other'),
            subreddit_topcategory.value.get(e[5], 'Other')))
    
    return (author[0], entities)
