build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y surfaces-cec-data || true
	rm -fr build dist *.egg-info

reinstall: uninstall install