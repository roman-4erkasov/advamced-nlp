.PHONY:*

PIP=venv/bin/pip3.10

install:
	python3 -m venv venv
	$(PIP) install -U pip
#	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt

clean:
	rm -rf venv

