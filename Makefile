build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y surfaces
	rm -fr build dist *.egg-info

install-dev:
	pip install -e ".[dev]"

install-test:
	pip install -e ".[test]"

install-build:
	pip install build

install-editable:
	pip install -e .

reinstall: uninstall install

reinstall-editable: uninstall install-editable

test-examples:
	cd tests; \
		python _test_examples.py

py-test:
	python -m pytest -x -p no:warnings tests/; \

test:  py-test test-examples

database:
	python -m collect_search_data.py
