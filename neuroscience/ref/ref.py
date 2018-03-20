import numpy as np
import os.path as op
import dipy.core.gradients as dpg
import nibabel as nib
from dipy.segment.mask import median_otsu
from download import download
from time import time


download()

#Load data
load_start = time()
img = nib.load('./data.nii.gz')
data = img.get_data()
affine = img.affine
img = None
print("Loading data: {0}s".format(time() - load_start))

#build mask
mask_start = time()
gtab = dpg.gradient_table('./bvals', './bvecs', b0_threshold=10)
mean_b0 = np.mean(data[..., gtab.b0s_mask], -1)
_, mask = median_otsu(mean_b0, 4, 2, False, vol_idx=np.where(gtab.b0s_mask), dilate=1)
print("Building mask: {0}s".format(time() - mask_start))

if not op.exists('./mask.nii.gz'):
    saving_mask_start = time()
    nib.save(nib.Nifti1Image(mask.astype(int), affine), 'mask.nii.gz')
    print("Saving mask: {0}s".format(time() - saving_mask_start))

# denoise
denoise_start = time()
from dipy.denoise import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

sigma = estimate_sigma(data)
denoised_data = nlmeans.nlmeans(data, num_threads=8, sigma=sigma, mask=mask)
print("Denoising: {0}s".format(time() - denoise_start))

if not op.exists('./denoised_data.nii.gz'):
    saving_denoise_start = time()
    nib.save(nib.Nifti1Image(denoised_data, affine), 'denoised_data.nii.gz')
    print("Saving denoise: {0}s".format(time() - saving_denoise_start))

#tensor model
model_start = time()
import dipy.reconst.dti as dti

ten_model = dti.TensorModel(gtab)
ten_fit = ten_model.fit(denoised_data, mask=mask)
print("Model fitting: {0}s".format(time() - model_start))

saving_model_start = time()
nib.save(nib.Nifti1Image(ten_fit.fa, affine), 'dti_fa.nii.gz')
nib.save(nib.Nifti1Image(ten_fit.md, affine), 'dti_md.nii.gz')
print("Saving model: {0}s".format(time() - saving_model_start))
