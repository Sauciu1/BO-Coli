
# Setup
To run the project, assuming you have python 3.13 installed:

### General
``` bash
pip install poetry
poetry lock
```

### UI

to run UI after installing the project.
From main dir

``` bash

poetry env activate
streamlit run src/UI_main.py
```


You bo-coli venv should now appear, run everything from it.


##
To deploy BOcoli with docker installed on system:
```powershell
git clone https://github.com/Sauciu1/BO-Coli
docker build -t bo_coli:v8 .


```

This will take a few minutes to compile, afterwards you can access the via your browser via.


```
docker run -p 8989:8989 bo_coli:v8
http://localhost:8989/
```