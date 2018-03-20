import os.path as op
import os
import boto3
import sys


def download(subject_id, output_directory='.'):
    boto3.setup_default_session(profile_name='hcp')
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('hcp-openaccess')

    for file_name in ['bvals', 'bvecs', 'data.nii.gz']:
        output_path = '{0}/{1}_{2}'.format(output_directory, subject_id, file_name)
        if op.exists(output_path):
            print("Using previously downloaded {0}".format(output_path))
        else:
            source_path = 'HCP/{0}/T1w/Diffusion/{1}'.format(subject_id, file_name)
            print("Downloading {0} to {1}".format(source_path, output_path))
            bucket.download_file(source_path, output_path)
            print("Done")


def remove(subject_id, output_directory='.'):
    for file_name in ['bvals', 'bvecs', 'data.nii.gz']:
        output_path = '{0}/{1}_{2}'.format(output_directory, subject_id, file_name)
        if op.exists(output_path):
            os.remove(output_path)


if __name__ == "__main__":
    for case_id in sys.argv[1:]:
        download(case_id)
