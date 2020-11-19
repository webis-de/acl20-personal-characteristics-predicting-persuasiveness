from pyspark import SparkContext, SparkConf, SparkFiles
import time
import pprint
import sys
import json

conf = SparkConf().setAppName("cmv")
sc = SparkContext(conf=conf)
pp = pprint.PrettyPrinter(width=100)

def json_loads(x):
    if x[-1] == ',':
        return json.loads(x[:-1])
    else:
        return json.loads(x)

def instance_of(x, instanceIds):
    if x is None:
        return False
    
    p31 = x.get('claims', {}).get('P31', {})
    
    for snak in p31:
        valueId = snak.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id', '')
        if valueId in instanceIds:
            return True

    return False

subclasses_human = ['Q5']

subclasses_org = []
with open('/home/username/data/subclasses_org.json', 'r') as fp:
    subclasses_org = json.load(fp)
    print(len(subclasses_org))
subclasses_org_bc = sc.broadcast(subclasses_org)

subclasses_geo = []
with open('/home/username/data/subclasses_geo.json', 'r') as fp:
    subclasses_geo = json.load(fp)
    print(len(subclasses_geo))
subclasses_geo_bc = sc.broadcast(subclasses_geo)


wikiFile = sc.textFile('/user/username/data/input/wikidata_repartitioned')
wikiData = wikiFile.filter(lambda x: '{' in x).map(lambda x : json_loads(x))
# print(wikiData.take(3))

human_objects = wikiData.filter(lambda x : instance_of(x, subclasses_human))
print("org_objects: " + str(org_objects.count()))
human_objects = geo_objects.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
human_objects.saveAsTextFile("hdfs://hadoop:8020/user/username/data/wikidata_human")

org_objects = wikiData.filter(lambda x : instance_of(x, subclasses_org_bc.value))
print("org_objects: " + str(org_objects.count()))
org_objects = org_objects.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
org_objects.saveAsTextFile("hdfs://hadoop:8020/user/username/data/input/wikidata_org")

geo_objects = wikiData.filter(lambda x : instance_of(x, subclasses_geo_bc.value))
print("geo_objects: " + str(geo_objects.count()))
geo_objects = geo_objects.map(lambda l : json.dumps(l, ensure_ascii=False).encode('utf8'))
geo_objects.saveAsTextFile("hdfs://hadoop:8020/user/username/data/input/wikidata_geo")

# geo_objects (subclasses3) - 4 230 273
# geo_objects (subclasses2) - 1 815 660
# organization (subclasses2) - 521 065