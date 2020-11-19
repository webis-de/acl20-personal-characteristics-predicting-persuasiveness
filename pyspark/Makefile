
help:
	@echo "clean - remove all build, test, coverage and Python artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "clean-test - remove test and coverage artifacts"
	@echo "lint - check style"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "build - package"

# all: default

# default: clean dev_deps deps test lint build

# .venv:
# 	if [ ! -e ".venv/bin/activate_this.py" ] ; then virtualenv --clear .venv ; fi

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr dist/
	# rm -f dist/jobs.zip dist/main.py

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

# deps: .venv
# 	. .venv/bin/activate && pip install -U -r requirements.txt -t ./src/libs
deps: pip install -U -r requirements.txt -t ./libs

# dev_deps: .venv
# 	. .venv/bin/activate && pip install -U -r dev_requirements.txt
dev_deps: pip install -U -r dev_requirements.txt

# lint:
# 	. .venv/bin/activate && pylint -r n src/main.py src/shared src/jobs tests

# test:
# 	. .venv/bin/activate && nosetests ./tests/* --config=.noserc

build: clean
	echo "build..."
	mkdir ./dist
	cp ./main.py ./dist	
	zip -q -r ./dist/jobs.zip ./jobs -x main.py -x=*._*
	# zip -q -r ./dist/shared.zip ./shared -x=*._*
	cd ./libs && zip -q -r ../dist/libs.zip . -x=*._*
	echo "done."
	# cd ./jobs && zip -x main.py -x \*libs\* -r ../dist/jobs.zip .
	#cd ./src/libs && zip -qr ../../dist/libs.zip .
	#cd ./src/libs && zip -rq ../../dist/libs.zip . -i .
