from __future__ import division, print_function

import numpy as np
import nibabel as nib

from itertools import product
from dipy.reconst.csdeconv import (auto_response, ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel)
from dipy.reconst.shm import (QballModel, CsaOdfModel, lazy_index)
from dipy.data import get_sphere
from dipy.reconst.peaks import peaks_from_model, reshape_peaks_for_visualization
from dipy.reconst.dti import TensorModel
from dipy.core.gradients import gradient_table
from dipy.viz import fvtk

def screenshot_odf(odf, sphere, filename, show=False):
    """Takes a screenshot of the odfs, saved as filename.png"""

    if ".png" not in filename:
        filename += '.png'

    ren = fvtk.ren()
    fodf_spheres = fvtk.sphere_funcs(odf, sphere, scale=1.8, norm=True)
    fvtk.add(ren, fodf_spheres)
 #   fvtk.add(ren, fvtk.axes())

    fodf_spheres.RotateZ(90)
    fodf_spheres.RotateX(180)
    fodf_spheres.RotateY(180)

    if show:
        fvtk.show(ren, size=(600, 600))

    fvtk.record(ren, out_path=filename, size=(1000, 1000))
    print('Saved illustration as', filename)


def screenshot_peaks(peaks_dirs, filename, peaks_values=None, show=False):
    """Takes a screenshot of the peaks, saved as filename.png"""

    if ".png" not in filename:
        filename += '.png'

    ren = fvtk.ren()
    fodf_peaks = fvtk.peaks(peaks_dirs, peaks_values, scale=1.8)
    fvtk.add(ren, fodf_peaks)
  #  fvtk.add(ren, fvtk.axes())

    fodf_peaks.RotateZ(90)
    fodf_peaks.RotateX(180)
    fodf_peaks.RotateY(180)

    if show:
        fvtk.show(ren, size=(600, 600))

    fvtk.record(ren, out_path=filename, size=(1000, 1000))
    print('Saved illustration as', filename)



# Get data from MASSIVE or simulations
# For every dataset
for num_dataset in range(1):
    ndir = 60
    shell = None
    directory = 'D:/H_schijf/Data/MASSIVE/Processed/'
    img = nib.load(directory + 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells_48_40_39.nii')
    data = img.get_data()
    affine = img.get_affine()

    bvals = np.loadtxt(directory + 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells.bval')
    bvecs = np.loadtxt(directory + 'DWIs_A_MD_C_native_rigid_2KS_ordered_shells.bvec')

    ind1000 = bvals < 1100
    data = data[..., ind1000]
    bvals = bvals[ind1000]
    bvecs = bvecs[:,ind1000]

    gtab = gradient_table(bvals=bvals, bvecs=bvecs, big_delta=51.6, small_delta=32.8,
                          b0_threshold=0.5)

    tag = 'Analysis/b1000'


    # Set parameters
    _where_b0s = lazy_index(gtab.b0s_mask)
    S0 = np.mean(data[:, :, :, _where_b0s])
    CSD_sh_order = 8
    Qball_sh_order = 8
    CSA_sh_order = 4
    sphere = get_sphere('symmetric724').subdivide()


    # Compute DTI LLS
    tenmodel = TensorModel(gtab, fit_method="LS")
    tenfit = tenmodel.fit(data)
    filename = directory + tag + '_DTILLS'
    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
    # Save everything
    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine()), '_evecs.nii.gz')
    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), img.get_affine()), '_evals.nii.gz')
    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), affine), filename + 'fa.nii.gz')

    # Compute DTI WLLS
    tenmodel = TensorModel(gtab, fit_method="WLS")
    tenfit = tenmodel.fit(data)
    filename = directory + tag + '_DTIWLLS'
    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
    # Save everything
    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine()), '_evecs.nii.gz')
    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), img.get_affine()), '_evals.nii.gz')
    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), affine), filename + 'fa.nii.gz')

    # Compute DTI NLLS
    tenmodel = TensorModel(gtab, fit_method="NLLS")
    tenfit = tenmodel.fit(data)
    filename = directory + tag + '_DTINLLS'
    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
    # Save everything
    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine()), '_evecs.nii.gz')
    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), img.get_affine()), '_evals.nii.gz')
    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), affine), filename + 'fa.nii.gz')

#    # Compute DTI RESTORE
#    tenmodel = TensorModel(gtab, fit_method="RT")
#    tenfit = tenmodel.fit(data)
#    filename = directory + tag + '_DTIRESTORE'
#    screenshot_odf(tenfit.odf(sphere), sphere, filename + "_odf.png", show=True)
#    screenshot_peaks(tenfit.evecs[:, :, :, :, 0], filename + "_peaks.png", tenfit.evals[:, :, :, 0], show=True)
#    # Save everything
#    nib.save(nib.Nifti1Image(tenfit.evecs.astype(np.float32), img.get_affine()), '_evecs.nii.gz')
#    nib.save(nib.Nifti1Image(tenfit.evals.astype(np.float32), img.get_affine()), '_evals.nii.gz')
#    nib.save(nib.Nifti1Image(tenfit.fa.astype(np.float32), affine), filename + 'fa.nii.gz')

    # Compute DTI REKINDLE


    # Compute CSD
    # Calibrate RF on whole or partial data?
    response = (np.array([0.0015, 0.0003, 0.0003]), S0)

    csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=CSD_sh_order)
    peaks_csd = peaks_from_model(model=csd_model,
                                 data=data,
                                 sphere=sphere,
                                 relative_peak_threshold=.25,
                                 min_separation_angle=25,
                                 parallel=False,
                                 npeaks=5,
                                 return_sh=True,
                                 normalize_peaks=False,
                                 return_odf=True,
                                 sh_order=CSD_sh_order)
    filename = directory + tag + '_CSD'

    nfib = np.sum(np.sum(np.abs(peaks_csd.peak_dirs), axis=-1) > 0, axis=-1).ravel()
    print("1 fiber", np.sum(nfib==1), "2 fibers", np.sum(nfib==2), "3 fibers", np.sum(nfib==3))

    screenshot_odf(peaks_csd.odf, sphere, filename + "_odf.png", show=True)
    screenshot_peaks(peaks_csd.peak_dirs, filename + "_CSDpeaks.png", peaks_csd.peak_values, show=True)
    # Save everything
    nib.save(nib.Nifti1Image(peaks_csd.shm_coeff.astype('float32'), affine), filename + 'fodf.nii.gz')
    nib.save(nib.Nifti1Image(reshape_peaks_for_visualization(peaks_csd), affine), filename + 'peaks.nii.gz')
    nib.save(nib.Nifti1Image(peaks_csd.peak_indices, affine), filename + 'fodf_peak_indices.nii.gz')


    # Compute Qball
    qball_model = QballModel(gtab, sh_order=Qball_sh_order)
    peaks_qball = peaks_from_model(model=qball_model,
                                   data=data,
                                   sphere=sphere,
                                   relative_peak_threshold=.25,
                                   min_separation_angle=25,
                                   parallel=False,
                                   npeaks=5,
                                   return_sh=True,
                                   normalize_peaks=False,
                                   return_odf=True,
                                   sh_order=Qball_sh_order)
    filename = directory + tag + '_Qball'
    screenshot_odf(peaks_qball.odf, sphere, filename + "_odf.png", show=True)
    screenshot_peaks(peaks_qball.peak_dirs, filename + "_peaks.png", peaks_qball.peak_values,
                     show=True)


    # Compute SDT
    # Calibrate RF on whole or partial data?
    ratio = 0.21197

    sdt_model = ConstrainedSDTModel(gtab, ratio=ratio, sh_order=Qball_sh_order)
    peaks_sdt = peaks_from_model(model=sdt_model,
                                   data=data,
                                   sphere=sphere,
                                   relative_peak_threshold=.25,
                                   min_separation_angle=25,
                                   parallel=False,
                                   npeaks=5,
                                   return_sh=True,
                                   normalize_peaks=False,
                                   return_odf=True,
                                   sh_order=Qball_sh_order)
    filename = directory + tag + '_SDT'
    screenshot_odf(peaks_sdt.odf, sphere, filename + "_odf.png", show=True)
    screenshot_peaks(peaks_sdt.peak_dirs, filename + "_peaks.png", peaks_sdt.peak_values,
                     show=True)


    # Compute CSA
    csa_model = CsaOdfModel(gtab, sh_order=CSA_sh_order)
    peaks_csa = peaks_from_model(model=csa_model,
                                 data=data,
                                 sphere=sphere,
                                 relative_peak_threshold=.25,
                                 min_separation_angle=25,
                                 parallel=False,
                                 npeaks=5,
                                 return_sh=True,
                                 normalize_peaks=False,
                                 return_odf=True,
                                 sh_order=CSA_sh_order)
    filename = directory + tag + '_CSA'
    screenshot_odf(peaks_csa.odf, sphere, filename + "_odf.png", show=True)
    screenshot_peaks(peaks_csa.peak_dirs, filename + "_peaks.png", peaks_csa.peak_values, show=True)


