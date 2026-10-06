.PHONY: all data emps hrtem co3o4 database clean

all: data database

data: emps hrtem co3o4

emps:
	python Scripts/download_EMPS.py

hrtem:
	python Scripts/download_HRTEM.py --demo

co3o4:
	python Scripts/download_Co3O4.py

database:
	python Scripts/create_database.py

clean:
	rm -f Data/Database/microscopy.db