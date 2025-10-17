setup:
	python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt
train:
	python -m src.train.pretrain
forecast:
	python -m src.infer.forecast DATA_PATH=$(DATA_PATH) H=$(H)
