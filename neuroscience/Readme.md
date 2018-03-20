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

## Ansible
A great way of deploying the app is by using [Ansible](http://docs.ansible.com/ansible/latest/index.html). 

Firstly you need to install Ansible locally. You can do it using [this tutorial](http://docs.ansible.com/ansible/latest/intro_installation.html).
Secondly prepare two files `hosts` and `aws_credentials`. First will tell Ansible to what servers deploy the app. 
Example of `hosts` file for aws ec2 servers:
```
[test_group]
18.196.69.101 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem
18.196.69.102 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem
18.196.69.103 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem
18.196.69.104 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem
```
IP-s at the beginning of each line represents server IP. tests.pem in the example is local path to pem key assigned to this instance. 

File `aws_credentials` should contain valid s3 hcp credentials. It will be copied to `~/.aws/credentials` on each 
deployed server.

After installing Ansible and preparing `hosts` and `aws_credentials` you can deploy the app on all servers using command:
```
ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i ./hosts ansible_setup.yml
```

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
