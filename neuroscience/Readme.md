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
[masters]
18.196.69.101 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem

[slaves]
18.196.69.102 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem
18.196.69.103 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem
18.196.69.104 ansible_user=ubuntu ansible_ssh_private_key_file=tests.pem
```
IP-s at the beginning of each line represents server IP. tests.pem in the example is local path to pem key assigned to this instance. 

For Azure cloud cluster this file may look as follows:
```
[masters]
master ansible_host=test.westus2.cloudapp.azure.com ansible_port=50001 ansible_user=ubuntu ansible_ssh_pass=PASS

[slaves]
slave1 ansible_host=test.westus2.cloudapp.azure.com ansible_port=50002 ansible_user=ubuntu ansible_ssh_pass=PASS
slave2 ansible_host=test.westus2.cloudapp.azure.com ansible_port=50003 ansible_user=ubuntu ansible_ssh_pass=PASS
slave3 ansible_host=test.westus2.cloudapp.azure.com ansible_port=50004 ansible_user=ubuntu ansible_ssh_pass=PASS
slave4 ansible_host=test.westus2.cloudapp.azure.com ansible_port=50005 ansible_user=ubuntu ansible_ssh_pass=PASS
slave5 ansible_host=test.westus2.cloudapp.azure.com ansible_port=50006 ansible_user=ubuntu ansible_ssh_pass=PASS
```
Note: inlining password in hosts file may not be safe and requires installing sshpass on local machine.

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

Test data consists of 3 files `bvals`, `bvecs` and `data.nii.gz`. Following command lists available cases:
```
aws s3 ls s3://hcp-openaccess/HCP/ --profile hcp
```


Utility `common/download.py` can be used to download them to project directory. Note: Files for single case weight in total 1.3GB.
```
python common/download.py 100307
```

# Running
## Local
```
python ref/main.py 100307
```

## Dask
Start dask scheduler on one machine with command:
```
dask-scheduler
```

On any number of other machines start worker process with command (replace SCHEDULER_IP with ip address of scheduler machine):
```
dask-worker SCHEDULER_IP:8786
```

Ansible playbooks `ansible_dask_start.yml` and `ansible_dask_stop.yml` automates starting dask cluster described in `hosts` file.
Starting dask on all machines of the cluster:
```
ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i ./hosts ansible_dask_start.yml
```

Stopping dask:
```
ANSIBLE_HOST_KEY_CHECKING=False ansible-playbook -i ./hosts ansible_dask_stop.yml
```

To run benchmark run following command on scheduler machine:
```
python dask/main.py SUBJECT_ID [SUBJECT_ID ...]
```

Note: If you forward port 8786 from scheduler machine to your machine you would be able to schedule tasks from your pc.
Forwarding ports can be achieved using `ssh`: `ssh -L 8786:127.0.0.1:8786 ...`.

## Spark
Start spark master process on one machine:
```
~/spark/sbin/start-master.sh
```

Start spark slaves on any machines:
```
~/spark/sbin/start-slave.sh spark://MASTER_IP:7077
```

Starting execution requires running on master machine:
```
python spark/main.py SUBJECT_ID [SUBJECT_ID ...]
```
