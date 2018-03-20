# Installation
## Virtual environment
```
virtualenv -p python venv  # just once to create python virtual environment
source venv/bin/activate
```
More information about virtual environments are available [here](https://virtualenv.pypa.io/en/stable/).

## Dependencies
```
pip install -r requirements.txt
```
Note: if you need to add any dependencies to the project please install them with `pip install ...` and remember to 
update the requirements file using `pip freeze > requirements.txt`. More information about `pip` tool can be found at
[https://pip.pypa.io/en/stable/user_guide/](https://pip.pypa.io/en/stable/user_guide/).

# Data
Data for the neuroscience use case is available at 
[HCP](https://wiki.humanconnectome.org/display/PublicData/How+To+Connect+to+Connectome+Data+via+AWS)

## Get the data from AWS S3
We assume that you have a file '.aws/credentials', 
that includes a section with credentials needed to access HCP data.
```
[hcp]
AWS_ACCESS_KEY_ID=XXXXXXXXXXXXXXXX
AWS_SECRET_ACCESS_KEY=XXXXXXXXXXXXXXXX
```

Test data consists of 3 files `bvals`, `bvecs` and `data.nii.gz`. Utility `ref/download.py` can be used to download them
to project directory. Note: They weight in total 1.3GB.
```
python ref/download.py
```

# Running
## Local
```
python ref/ref.py
```
