.PHONY:*

PIP=ve/bin/pip
PY=ve/bin/python

install:
	python3.8 -m venv ve
	${PIP} install -U pip
	${PIP} install -r requirements.txt

get_fasttext:
	# wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz
	# gzip -cd cc.ru.300.vec.gz > data/cc.ru.300.vec
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz -P ./data/
	gzip -cd ./data/cc.ru.300.bin.gz > data/cc.ru.300.bin
	rm data/cc.ru.300.bin.gz

