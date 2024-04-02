.PHONY:*

PIP=./venv/bin/pip3.10
PY=./venv/bin/python

get_data:
	mkdir -p ./data
	wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz
	gzip -cd lenta-ru-news.csv.gz > ./data/lenta-ru-news.csv


prepare_data: get_data
	${PY} prepare_data.py

install: clean
	python3 -m venv venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf venv

