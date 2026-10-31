.PHONY: all data analysis charts test notebook clean

PYTHON ?= python3
JUPYTER ?= jupyter

all: data analysis test notebook

data:
	@echo "--> Generating FMEA, experimental runs, and SQLite database..."
	$(PYTHON) src/generate_data.py

analysis:
	@echo "--> Running OLS regression, ANOVA decomposition, and design space optimization..."
	$(PYTHON) analysis.py

charts: analysis
	@echo "--> Charts generated in charts/ directory."

test:
	@echo "--> Running verification self-checks..."
	$(PYTHON) tests/selfcheck.py

notebook:
	@echo "--> Executing analysis.ipynb in-place..."
	$(JUPYTER) nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 analysis.ipynb

clean:
	@echo "--> Cleaning output artifacts..."
	rm -rf out/* charts/* data/doe.db data/*.csv
