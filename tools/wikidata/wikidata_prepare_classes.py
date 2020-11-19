### Prepare wikidata entitity classes lists:
### human Q5 1 class

### organization Q43229 + indirect subclasses (3 level deep)
### facility Q13226383
### university Q3918
### exclude: state (Q7275)
### org: exclude city Q515, country Q6256, ...? political territorial entity (Q1048835), administrative territorial entity (Q56061)

### geographical object Q618123 + indirect subclasses (2 level deep)
### geographic location Q2221906 + indirect subclasses (2 level deep)
### geographic region Q82794 + indirect subclasses (2 level deep)
### city Q515
### country Q6256
### exclude facility (Q13226383), organization Q43229
### geo: exclude university Q3918, organization Q43229, ...?


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

def subclass_of(x, classes, exclude = []):
    if x is None:
        return False
    
    p31 = x.get('claims', {}).get('P279', {})
    
    for snak in p31:
        valueId = snak.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id', '')
        if valueId in classes and valueId not in exclude:
            return True

    return False

start = time.time()

# wikiFile = sc.textFile('/user/lusername/data/input/latest-all.json.bz2')
wikiFile = sc.textFile('/user/lusername/data/input/wikidata_repartitioned')
wikiData = wikiFile.filter(lambda x: '{' in x).map(lambda x : json_loads(x))

print('working ... \n')

###
### org
###
subclasses = []
subclasses.append(['Q43229'])
subclasses_bc = sc.broadcast(subclasses)
for i in range(1,4):
    subclasses_obj = wikiData.filter(lambda x : subclass_of(x, subclasses_bc.value[i-1], ['Q515','Q6256', 'Q7275', 'Q1048835', 'Q56061']))
    subclasses_list = subclasses_obj.map(lambda x: x['id'] ).collect()
    print("Level #" + str(i) + ":" + str(len(subclasses_list)))
    subclasses.append(subclasses_list)
    subclasses_bc = sc.broadcast(subclasses)

subclasses_flat = [item for sublist in subclasses for item in sublist]
subclasses_flat.extend(['Q13226383', 'Q3918'])

print("Org subclasses total:" + str(len(subclasses_flat)))
with open('subclasses_org3.json', 'w') as fp:
    json.dump(subclasses_flat, fp)

###
### geo
###
subclasses = []
subclasses.append(['Q618123', 'Q2221906', 'Q82794'])
subclasses_bc = sc.broadcast(subclasses)
for i in range(1,3):
    subclasses_obj = wikiData.filter(lambda x : subclass_of(x, subclasses_bc.value[i-1], ['Q13226383', 'Q43229', 'Q3918']))
    subclasses_list = subclasses_obj.map(lambda x: x['id'] ).collect()
    print("Level #" + str(i) + ":" + str(len(subclasses_list)))
    subclasses.append(subclasses_list)
    subclasses_bc = sc.broadcast(subclasses)

subclasses_flat = [item for sublist in subclasses for item in sublist]
subclasses_flat.extend(['Q515','Q6256'])

print("Geo subclasses total:" + str(len(subclasses_flat)))
with open('subclasses_geo3.json', 'w') as fp:
    json.dump(subclasses_flat, fp)

# 86
# 556
# 6079

end = time.time()
print(end - start, 'seconds')


