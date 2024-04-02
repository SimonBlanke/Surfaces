build:
	python -m build

install: build
	pip install dist/*.whl

install-requirements:
	python -m pip install -r ./requirements/requirements.txt

install-test-requirements:
	python -m pip install -r ./requirements/requirements-test.txt

dev-install:
	pip install -e .

reinstall:
	pip uninstall -y surfaces
	rm -fr build dist surfaces.egg-info
	make install

tox-test:
	tox -- -x -p no:warnings -rfEX tests/ \

py-test:
	python -m pytest -x -p no:warnings tests/; \

test:  py-test tox-test

requirement:
	cd requirements/; \
		pip-compile requirements.in;\
		pip-compile requirements-test.in

database:
	python -m collect_search_data.py