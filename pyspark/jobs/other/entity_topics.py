'''
- load categories list into a graph
- mark second-level categories - categories to find shortest path from entity pages
- load entity list (flat, reduce author_entity dataset)
- for each entity (page id) find closest parent category
- todo: compute shortest paths to 2nd-level categories
'''

from pyspark import SparkContext, SparkConf, SparkFiles

import networkx
import json
import pprint
import csv
import pickle
import sys

def process_page(page, g, topics):
    page_topics = {}
    p_id = page[0]
    filtered = page[1]

    # compute shortest paths from each category of a page to each of main wikipedia topics
    shortest_paths = {}
    for scat_id, scat_title in filtered:
        source_cat_shortest_paths = []
        for topic in topics.value:
            try:
                source_cat_shortest_paths.append((topic[0], topic[1], nx.shortest_path_length(g.value, source=scat_id, target=topic[0])))
            except Exception as e:
                continue
        sorted_paths = sorted(source_cat_shortest_paths, key=lambda x: x[2])
        min_paths = [c for c in sorted_paths if c[2] == sorted_paths[0][2]] # for each category get all topics with minimal path
        shortest_paths[(scat_id, scat_title)] = min_paths

    # group topics among all categories of a page and count frequency for each topic
    topics_reduced = {}
    for sc, v in shortest_paths.items():
        for id, topic, count in v:
            if topic not in topics_reduced:
                topics_reduced[topic] = 1
            else:
                topics_reduced[topic] += 1

    return (p_id, topics_reduced)

conf = SparkConf().setAppName("entity_topics")
sc = SparkContext(conf=conf)
pp = pprint.PrettyPrinter(width=100)

sc.addPyFile('/home/username/src/pyspark/dist/libs.zip')
sys.path.insert(0, SparkFiles.get('/home/username/src/pyspark/dist/libs.zip'))
import networkx

_topics = [(693763, 'Academic disciplines'),
    (4892515, 'Arts'),
    (771152, 'Business'),
    (24980271, 'Concepts'),
    (694861, 'Culture'),
    (696763, 'Education'),
    (693016, 'Entertainment'),
    (2766046, 'Events'),
    (693800, 'Geography'),
    (751381, 'Health'),
    (693555, 'History'),
    (1004110, 'Humanities'),
    (8017451, 'Language'),
    (691928, 'Law'),
    (2389032, 'Life'),
    (690747, 'Mathematics'),
    (696603, 'Nature'),
    (691008, 'People'),
    (691810, 'Philosophy'),
    (695027, 'Politics'),
    (722196, 'Reference'),
    (692694, 'Religion'),
    (691182, 'Science'),
    (1633936, 'Society'),
    (693708, 'Sports'),
    (696648, 'Technology'),
    (48005914, 'Universe'),
    (3260154, 'World')]
topics = sc.broadcast(_topics)

print('Loading categorytree...')
cats = pickle.load(open('/home/username/data/wikipedia/wikipedia-categorytree-corrected.pickle', 'rb'))

# fix Universe topic
cats[48005914] = {'id': 48005914, 'title': 'Universe', 'parents': [], 'pages': []}
universe_children = [865456, 1926287, 696603, 2217510, 744586]
for child in universe_children:
    cats[child]['parents'].append(48005914)

# create graph from the category list
print('Creating graph...')
import networkx as nx
_g = nx.DiGraph()
for k,v in cats.items():
    _g.add_node( (v['id'], v['title']) )
    for parent in v['parents']:
        _g.add_edge(v['id'], parent)
g = sc.broadcast(_g)

# create page->categories dictionary
print('Creating page-categories dict...')
page_cats = {}
for k,v in cats.items():
    for page in v['pages']:
        if page not in page_cats:
            page_cats[page] = set()
        page_cats[page].update( [(v['id'], v['title'])] )

# process each page
print('Process pages...')
page_cats_rdd = sc.parallelize(page_cats.items())
page_topics = page_cats_rdd.map(lambda x: process_page(x, g, topics)).collect()


print('Saving pickle...')
pickle.dump(page_topics, open('/home/username/data/output/_jobs/page_topics.pickle', 'wb'))
