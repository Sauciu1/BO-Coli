
# Setup
To run the project, assuming you have python 3.13 installed:

### General
```cmd
pip install poetry
poetry lock
poetry install
poetry env activate
```

You bo-coli venv should now appear, run everything from it.


To run what I need for HPC is :
```cmd 
conda create -n "bo_coli" python=3.13.0 ipython
conda activate bo_coli
conda install pip
pip install poetry
poetry install --no-root
poetry run python h6_simulations.py


```y
