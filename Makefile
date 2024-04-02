build:
	python -m build

install: build
	pip install dist/*.whl

install-requirements:
	python -m pip install ./requirements/requirements.in

install-test-requirements:
	python -m pip install ./requirements/requirements-test.in

install-build-requirements:
	python -m pip install ./requirements/requirements-build.in

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
		pip-compile requirements-test.in;\
		pip-compile requirements-build.in

database:
	python -m collect_search_data.py