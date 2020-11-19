import argparse
import logging
import operator
import re
import os.path
import json

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

### Configure command line argument parser
parser = argparse.ArgumentParser(
    description="Get categories' subcategories and pages from the Wikipedia XML dump.")
parser.add_argument(
    "-s", "--print-schema", action="store_true",
    help="Show the dataframe schema inferred by the XML parser.")
parser.add_argument(
    "dump_location", help="HDFS path to the XML dump.")
parser.add_argument(
    "output_location", help="HDFS path to the output directory.")

args = parser.parse_args()

# The output directory on the HDFS
output_location = str(args.output_location) if args.output_location else ""

### Configure Spark
logging.info("Configuring Spark")
sparkConf = SparkConf().setAppName("PySpark Wikipedia Cat/Subcat Processer")
sc = SparkContext(conf=sparkConf)
sqlC = SQLContext(sc) 

logging.info("Loading XML dump from " + args.dump_location)

### Load the Wikipedia dump using spark-xml
wikiDf = (sqlC.read.format('com.databricks.spark.xml')
          .options(rowTag='page')
          .load(args.dump_location))

if args.print_schema:
    logging.info("Inferred DataFrame schema from XML dump")
    wikiDf.printSchema()

### Get regular pages
# Filters: 
### ns=0: (Namespace 0 = regular page)
### redirect.isNull: we only want pages that have content, not redirect pages
#wikiRegularPages = wikiDf.where(wikiDf.ns == 0).where(wikiDf.redirect.isNull())

### Get Category pages
# Filters:
### ns=14: (Namespace 14 = category page)
### redirect.isNull: we only want pages that have content, not redirect pages
wikiCategoryPages = wikiDf.where(wikiDf.ns == 14).where(wikiDf.redirect.isNull())
# total_cats_subcats = wikiCategoryPages.count()
# logging.info("Found {} categories and subcategories in dump.".format(total_cats_subcats))


##################################################################
# Broadcast categories (id, names)
logging.info("Broadcasting category info")
broadcastCategories = sc.broadcast( wikiCategoryPages.rdd.map(lambda r: (r.title, r.id)).collectAsMap() )
logging.info("-- done")
##################################################################

# Process Pages
logging.info("Processing the xml dump")

def process_page(page, page_type):
    output = {
        "id": "",
        "title": "",
        "categories": []
    }

    try:
        # Page id
        output["id"] = page.id

        # Page title
        output['title'] = page.title.replace('Category:', '')

        # Get latest revision
        p_latest_revision_text = page.revision.text._VALUE

        # if p_text and p_text is not None:
        if p_latest_revision_text and p_latest_revision_text is not None:
            # Get the categories in this page
            subcategories = re.findall("\[\[(Category:.*?)\]\]", p_latest_revision_text)
            if subcategories:
                for (cat) in subcategories:
                    # cat title
                    final_cat_title = cat.split('|')[0]

                    # get the category id from the broadcasted categories
                    final_cat_id = broadcastCategories.value.get(final_cat_title, '')

                    # Add to the list of categories
                    if final_cat_id and final_cat_id is not None:
                        output["categories"].append(final_cat_id)

    except AttributeError as error:
        pass

    # For category pages, return a tuple
    # For regular pages, return an array (because we need to map on it again later, it's easier)
    if page_type == 'category':
        values = [output['title'], output['categories']]
        return (output['id'], tuple(values))

    else:
        return output

rdd_regular_pages = wikiRegularPages.rdd.map(lambda r: process_page(r, "page"))

##################################################################
# Transform the data from every pages having categories to every category having pages
logging.info("Fetching the list of pages for every category")

def get_pages_of_categories(page, category_id):
    return (category_id, (page['id']))

rdd_regular_pages = rdd_regular_pages.filter(lambda x: len(x['categories']) != 0)
rdd_cat_pages = rdd_regular_pages.flatMap(lambda page: [get_pages_of_categories(page, category_id) for category_id in page['categories']]).groupByKey().mapValues(list)

# print(rdd_cat_pages.take(5))
logging.info("-- done")
# output: [(123, ([5755382])), (8787, ([5755382]))]
##################################################################

# Process category pages
logging.info("Fetching the list of subcats of categories")
rdd_cat_subcats = wikiCategoryPages.rdd.map(lambda r: process_page(r, "category"))

# print(rdd_cat_subcats.take(5))
logging.info("-- done")
##################################################################

# MERGE the two RDDs
logging.info("Merging the two RDDs")
joined_rdd = rdd_cat_subcats.join(rdd_cat_pages)
logging.info("-- done")

# Flatten the results
def flatten(foo):
    for x in foo:
        if hasattr(x, '__iter__') and not isinstance(x, str) and not isinstance(x, list):
            for y in flatten(x):
                yield y
        else:
            yield x

logging.info("Flattening the results")
flattened_results = joined_rdd.map(lambda row: tuple(flatten(row)))

# print(flattened_results.take(10))
logging.info("-- done")
##################################################################

# Map to JSON
logging.info("Mapping to json")

final_results = flattened_results.map(lambda row: {'id': row[0], 'title': row[1].replace('Category:',''), 'subcategories': row[2], 'pages': row[3]})
finalResultsJSON = final_results.map(json.dumps)

logging.info("-- done")
print(finalResultsJSON.take(5))

##################################################################

# save to HDFS
logging.info("Saving output " + str(output_location))
finalResultsJSON.saveAsTextFile(output_location)
logging.info("-- done")
