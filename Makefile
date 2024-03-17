dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

dev-install:
	pip install -e .

reinstall: requirement
	pip uninstall -y surfaces
	rm -fr build dist surfaces.egg-info
	python setup.py bdist_wheel
	pip install dist/*

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