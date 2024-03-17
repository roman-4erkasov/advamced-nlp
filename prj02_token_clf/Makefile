.PHONY:*

PIP=venv/bin/pip

install:
	python3 -m venv venv
	$(PIP) install -U pip
	$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install -r requirements.txt


