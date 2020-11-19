import re
from os.path import join, dirname
from tempfile import NamedTemporaryFile
from glob import glob
from os.path import basename

from nose.tools import assert_equal, assert_multi_line_equal, assert_true

from semanticizest import Semanticizer
from semanticizest._semanticizer import create_model

tempfile = NamedTemporaryFile()
db = create_model(join(dirname(__file__),
                       'nlwiki-20140927-pages-articles-sample.xml'),
                  tempfile.name)
sem = Semanticizer(tempfile.name)


def test_semanticizer():
    text = """Aangezien de aarde een planeet is, kunnen de aardwetenschappen
ook als een tak van de planetologie beschouwd worden. Aardwetenschappelijke
kennis, met name geomorfologie, wordt bijvoorbeeld ook toegepast voor de
zoektocht naar sporen van water, sneeuw en ijs op de planeet Mars."""
    tokens = re.split(r'\W+', text)

    expected = set(['Planeet', 'Planetologie', 'Kennis (wetenschap)',
                    'Geomorfologie', 'Mars (planeet)'])
    concepts = set(string for _, _, string, _ in sem.all_candidates(tokens))

    assert_equal(expected, concepts)


def test_semanticizer_redirect():
    text = """In 1902 werkte hij even bij een Architekt."""
    tokens = re.split(r'\W+', text)

    expected = set(['Architect'])
    actual = set(string for _, _, string, _ in sem.all_candidates(tokens))

    assert_equal(expected, actual)


def test_semanticizer_nlwiki():
    tempfile = NamedTemporaryFile()
    db = create_model(join(dirname(__file__),
                           'nlwiki-20140927-pages-articles-sample.xml'),
                      tempfile.name)
    sem = Semanticizer(tempfile.name)

    dirs = {d: join(dirname(__file__), 'nlwiki', d)
            for d in "in expected actual".split()}

    input_test_cases = glob(join(dirs['in'], '*'))
    assert_equal(len(input_test_cases), 20,
                 msg=("number of input test cases in %r should be 20"
                      % dirs['in']))

    for doc in input_test_cases:
        fname = basename(doc)
        with open(doc) as f:
            with open(join(dirs['actual'], fname), 'w') as out:
                tokens = f.read().split()
                out.write("\n".join(str(cand)
                                    for cand in sem.all_candidates(tokens)))
        with open(join(dirs['expected'], fname)) as f:
            expected = f.read()
        with open(join(dirs['actual'], fname)) as f:
            actual = f.read()

        assert_multi_line_equal(expected,
                                actual)


def test_semanticizer_nlwiki_no_ngrams():
    tempfile = NamedTemporaryFile()
    db = create_model(join(dirname(__file__),
                           'nlwiki-20140927-pages-articles-sample.xml'),
                      tempfile.name, N=None)
    sem = Semanticizer(tempfile.name)

    assert_true(True)
