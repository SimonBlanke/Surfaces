install:
	python -m pip install -r ./requirements/requirements.txt

install-test:
	python -m pip install -r ./requirements/requirements-test.txt

dev-install:
	pip install -e .

reinstall:
	pip uninstall -y surfaces
	rm -fr build dist surfaces.egg-info
	python -m build
	pip install dist/*.whl

tox-test:
	tox -- -x -p no:warnings -rfEX tests/ \

test-pytest:
	python -m pytest -x -p no:warnings tests/; \

test:  test-pytest tox-test

requirement:
	cd requirements/; \
		pip-compile requirements.in;\
		pip-compile requirements-test.in

database:
	python -m collect_search_data.py