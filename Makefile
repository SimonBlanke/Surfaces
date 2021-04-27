dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y gradient_free_objective_functions
	rm -fr build dist gradient_free_objective_functions.egg-info
	python setup.py bdist_wheel
	pip install dist/*
