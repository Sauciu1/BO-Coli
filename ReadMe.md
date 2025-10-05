
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
To deploy BOcoli with docker:
```powershell
docker build -t bo_coli:v6 .
docker run -p 8989:8989 bo_coli:v6

```

You can now connect to the app via your browser window at:

```
http://localhost:8989/
```