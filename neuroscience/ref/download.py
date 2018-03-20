import os.path as op
import boto3


def download():
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')

    data_files = {'./bvals': 'HCP/994273/T1w/Diffusion/bvals',
                  './bvecs': 'HCP/994273/T1w/Diffusion/bvecs',
                  './data.nii.gz': 'HCP/994273/T1w/Diffusion/data.nii.gz'}

    for k in data_files.keys():
        if op.exists(k):
            print("File {0} already downloaded".format(k))
        else:
            print("Downloading {0} to {1}".format(data_files[k], k))
            bucket.download_file(data_files[k], k)
            print("Done")


if __name__ == "__main__":
    download()
