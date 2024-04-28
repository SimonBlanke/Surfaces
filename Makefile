build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y surfaces
	rm -fr build dist *.egg-info

install-requirements:
	python -m pip install -r ./requirements/requirements.in

install-test-requirements:
	python -m pip install -r ./requirements/requirements-test.in

install-build-requirements:
	python -m pip install -r ./requirements/requirements-build.in

install-editable:
	pip install -e .

reinstall: uninstall install

reinstall-editable: uninstall install-editable

test-examples:
	cd tests; \
		python _test_examples.py

tox-test:
	tox -- -x -p no:warnings -rfEX tests/ \

py-test:
	python -m pytest -x -p no:warnings tests/; \

test:  py-test test-examples

requirement:
	cd requirements/; \
		pip-compile requirements.in;\
		pip-compile requirements-test.in;\
		pip-compile requirements-build.in

database:
	python -m collect_search_data.py