# Intro 
Welcome to BO-Coli (Bayesian Optimisation for E Coli), the software tool developed for no-code Bayesian optimisation of biological systems to support Imperial's iGEM 2025.

There are two main parts to this repository:
* The BO-Coli software, which can be run for simple shell commands and accessed from within your browser. It is self-encompassing and requires no further interactions with the repository.

* The supporting simulations and notebooks for the custom GP kernel developed for technical repeats and heteroscedastic noise can be found within the ./notebooks/ folder.




# Software
The package provides a no-code solution for running Bayesian Experiments. All instructions and description can be found within the applet itself and the accompanying iGEM pages.

### Deployment

To deploy BOColi, you first need [docker](https://www.docker.com/) and [git](https://git-scm.com/downloads) installed on your system:
**CHANGE THIS TO WHERE IS IT STORED ON THE WIKI**
``` bash
git clone https://github.com/Sauciu1/bo-coli
cd bo-coli
docker build -t bo_coli:v9 .
```

This will take ~10 minutes to compile, afterwards it can be rapidly deployed and accessed from your browser:

``` bash
docker run -p 8989:8989 bo_coli:v9
```

The link to the applet for your browser is then gonna be.
``` bash
http://localhost:8989/
```


# Simulations and the notebook

Most important data is found at[.\notebooks\batch_bayesian_test.ipynb](.\notebooks\batch_bayesian_test.ipynb)


To run the project, locally you will need python 3.13 installed on the system, from within project folder:


``` bash
pip install poetry
poetry lock
poetry install
poetry env activate
```

#### Running the software without docker
You can run the 

``` bash
poetry run streamlit run ./src/ui/main_ui.py --server.port=8989 --server.address=0.0.0.0
```
It will once again be accessible via browser at:

``` bash
http://localhost:8989/
```


I hope these can be useful to someone, if you run into any issues or need some help, please reach out to me on GitHub.
Best,
Povilas
Dry lab lead on iGEM 