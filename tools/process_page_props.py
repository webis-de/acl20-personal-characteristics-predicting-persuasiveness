import sqlite3
import re
import xml.etree.ElementTree as etree   # don't use LXML, it's slower (!)
import logging
import six
import gzip
import sys

def extract_pages(f):
    """Extract pages from Wikimedia database dump.

    Parameters
    ----------
    f : file-like or str
        Handle on Wikimedia article dump. May be any type supported by
        etree.iterparse.

    Returns
    -------
    pages : iterable over `Page`s
        namedtuples containging the fields (page_id, title, content,
        redirect_target) triples.  In Python 2.x, may produce either
        str or unicode strings.

    """
    elems = etree.iterparse(f, events=["end"])

    # We can't rely on the namespace for database dumps, since it's changed
    # it every time a small modification to the format is made. So, determine
    # those from the first element we find, which will be part of the metadata,
    # and construct element paths.
    _, elem = next(elems)
    namespace = _get_namespace(elem.tag)
    ns_mapping = {"ns": namespace}
    ns_path = "./{%(ns)s}ns" % ns_mapping
    page_tag = "{%(ns)s}page" % ns_mapping
    text_path = "./{%(ns)s}revision/{%(ns)s}text" % ns_mapping
    id_path = "./{%(ns)s}id" % ns_mapping
    title_path = "./{%(ns)s}title" % ns_mapping
    redir_path = "./{%(ns)s}redirect" % ns_mapping

    for _, elem in elems:
        if elem.tag == page_tag:
            if elem.find(ns_path).text != '0':
                continue

            text = elem.find(text_path).text
            if text is None:
                # Empty article; these occur in Wikinews dumps.
                continue
            redir = elem.find(redir_path)
            redir = (_tounicode(redir.attrib['title'])
                     if redir is not None else None)

            text = _tounicode(text)
            title = _tounicode(elem.find(title_path).text)

            yield Page(int(elem.find(id_path).text), title, text, redir)

            # Prune the element tree, as per
            # http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
            # We do this only for <page>s, since we need to inspect the
            # ./revision/text element. That shouldn't matter since the pages
            # comprise the bulk of the file.
            elem.clear()

    for i, page in enumerate(extract_pages(f), 1):
        if i % 10000 == 0:
            _logger.info("%d articles done", i)
        if page.redirect is not None:
            redirects[page.title] = page.redirect
            continue

def _open(f):
    if isinstance(f, six.string_types):
        if f.endswith('.gz'):
            return gzip.open(f)
        elif f.endswith('.bz2'):
            return BZ2File(f)
        return open(f)
    return f

def page_properties(c, page_id):
    defaults = {'page_id': page_id, 'defaultsort': 'NULL', 'wikibase_item': 'NULL', 'wikibase_shortdesc': 'NULL'}
    try:
        pp = c.execute('''select pp_page, pp_propname, pp_value
                          from page_props
                          where pp_propname in ('defaultsort', 'wikibase_item', 'wikibase_shortdesc') 
                          and pp_page = ?''', (int(page_id),))
    except sqlite3.OperationalError as e:
        print("exception with page_id: {0}".format(page_id))
        return False

    res = pp.fetchall()
    properties = {}
    if (len(res)):
        properties = dict((b, c) for a, b, c in res)

    return {**defaults, **properties}

logger = logging.getLogger('semanticizest')
logger.addHandler(logging.StreamHandler(sys.stderr))
logger.setLevel('INFO')

page_props_file = "prop.db"
logger.info("Open page-props database at %r" % page_props_file)
try:
    db_props = sqlite3.connect(page_props_file)
except sqlite3.OperationalError as e:
    if 'unable to open' in str(e):
        die("%s: %r" % (e, page_props_file))
    else:
        raise

dump = "wikipedia/enwiki.xml.bz2"
f = _open(dump)
for i, page in enumerate(extract_pages(f), 1):
    if i % 10000 == 0:
        _logger.info("%d articles done", i)
    if page.redirect is not None:
        redirects[page.title] = page.redirect
        continue

    properties = page_properties(db_props, page.page_id)
    print(properties)
    system.exit(1)
    # if (properties):
    #     c.execute('''insert into pages values (?, ?, ?, ?)''',
    #               (page.page_id, page.title, properties['wikibase_item'], properties['wikibase_shortdesc']))

    # db.commit()

### todo: run through all targets and fill page_id and other properties

