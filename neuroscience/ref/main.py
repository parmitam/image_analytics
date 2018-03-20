import numpy as np
import os.path as op
import dipy.core.gradients as dpg
import nibabel as nib
import sys
from dipy.segment.mask import median_otsu
from common.download import download
from time import time
import datetime


case_id = 100307
if len(sys.argv) >= 2:
    case_id = sys.argv[1]

print("[{0}] Downloading data for case: {1}".format(datetime.datetime.now(), case_id))
download(case_id)
file_prefix = './{0}_'.format(case_id)
print("[{0}] Download finished".format(datetime.datetime.now()))


print("[{0}] Loading data".format(datetime.datetime.now()))
load_start = time()
img = nib.load(file_prefix + 'data.nii.gz')
data = img.get_data()
affine = img.affine
img = None
print("[{0}] Loading done in {0}s".format(datetime.datetime.now(), time() - load_start))


print("[{0}] Building mask".format(datetime.datetime.now()))
mask_start = time()
gtab = dpg.gradient_table(file_prefix + 'bvals', file_prefix + 'bvecs', b0_threshold=10)
mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)
_, mask = median_otsu(mean_b0, 4, 2, False, vol_idx=np.where(gtab.b0s_mask), dilate=1)
print("[{0}] Built mask {1}".format(datetime.datetime.now(), time() - mask_start))

mask_file = file_prefix + 'mask.nii.gz'
if not op.exists(mask_file):
    print("[{0}] Saving mask".format(datetime.datetime.now()))
    saving_mask_start = time()
    nib.save(nib.Nifti1Image(mask.astype(int), affine), mask_file)
    print("[{0}] Saved mask to {2} in {1}s".format(datetime.datetime.now(), time() - saving_mask_start, mask_file))
else:
    print("[{0}] Skipping mask save - already exists".format(datetime.datetime.now()))


print("[{0}] Denoising".format(datetime.datetime.now()))
denoise_start = time()
from dipy.denoise import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

sigma = estimate_sigma(data)
denoised_data = nlmeans.nlmeans(data, num_threads=8, sigma=sigma, mask=mask)
print("[{0}] Finished denoising in {1}s".format(datetime.datetime.now(), time() - denoise_start))

denoised_file = file_prefix + 'denoised_data.nii.gz'
if not op.exists(denoised_file):
    print("[{0}] Saving denoising result".format(datetime.datetime.now()))
    saving_denoise_start = time()
    nib.save(nib.Nifti1Image(denoised_data, affine), denoised_file)
    print("[{0}] Saved denoised file to {1} in {2}s".format(datetime.datetime.now(), denoised_file, time() - saving_denoise_start))
else:
    print("[{0}] Skipping denoised file save - already exists".format(datetime.datetime.now()))


#tensor model
model_start = time()
import dipy.reconst.dti as dti

ten_model = dti.TensorModel(gtab)
ten_fit = ten_model.fit(denoised_data, mask=mask)
print("Model fitting: {0}s".format(time() - model_start))

saving_model_start = time()
nib.save(nib.Nifti1Image(ten_fit.fa, affine), file_prefix + 'dti_fa.nii.gz')
nib.save(nib.Nifti1Image(ten_fit.md, affine), file_prefix + 'dti_md.nii.gz')
print("Saving model: {0}s".format(time() - saving_model_start))
