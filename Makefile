.PHONY: all data emps hrtem co3o4 database pipeline check clean
all: emps database

data: emps hrtem co3o4
emps:
	python Scripts/download_EMPS.py
hrtem:
	python Scripts/download_HRTEM.py --demo
co3o4:
	python Scripts/download_Co3O4.py
database:
	python Scripts/create_database.py
pipeline: all
	python src/tensors.py
check:
	python src/check_tensors.py
clean:
	rm -f Data/Database/microscopy.db data/inputs/*.safetensors data/inputs/manifest.json

# Model & Report Part I: ETL -> tensors -> validation -> training -> results -> PDF
.PHONY: submission
submission: pipeline
	python src/check_tensors.py
	python src/train_report.py
	test -s artifacts/results.json && test -s report.pdf
