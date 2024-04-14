.PHONY:*

PIP=./venv/bin/pip3.10
PY=./venv/bin/python


get_lenta_data:
	mkdir -p ./data
	wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz -P data/
	gzip -cd data/lenta-ru-news.csv.gz > ./data/lenta-ru-news.csv


#get_topic_modeling_data:
#	mkdir -p ./data
#	wget http://vectors.nlpl.eu/repository/11/195.zip -P data/
#	mkdir data/topic_modeling
#	unzip data/195.zip -d data/topic_modeling/


prepare_data: #get_lenta_data
	${PY} src/preprocessing.py

install: clean
	python3 -m venv venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

clean:
	rm -rf venv

