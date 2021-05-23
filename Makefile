dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y bbox_functions
	rm -fr build dist bbox_functions.egg-info
	python setup.py bdist_wheel
	pip install dist/*
