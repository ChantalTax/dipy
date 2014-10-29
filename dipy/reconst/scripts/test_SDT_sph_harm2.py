# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 19:16:54 2014

@author: user
#"""

import warnings
import numpy as np
import numpy.testing as npt
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           run_module_suite)
from dipy.data import get_sphere, get_data
from dipy.sims.voxel import (multi_tensor,
                             single_tensor,
                             multi_tensor_odf,
                             single_tensor_odf,
                             all_tensor_evecs,
                             single_tensor_qb_odf)
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel,
                                   forward_sdeconv_mat,
                                   odf_sh_to_sharp,
                                   auto_response, recursive_response)
from dipy.reconst.peaks import peak_directions
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.shm import (sf_to_sh, sh_to_sf, QballModel,
                              CsaOdfModel, sph_harm_ind_list,real_sph_harm)
from dipy.viz import fvtk
from dipy.reconst.shm import (lazy_index,
                              real_sym_sh_basis)
from dipy.core.geometry import cart2sphere
import dipy.reconst.dti as dti
from dipy.reconst.dti import fractional_anisotropy
from dipy.core.sphere import Sphere

from multiprocessing import freeze_support

if __name__=='__main__':
    freeze_support()


    SNR = None
    S0 = 1
    sh_order = 8

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)
    sphere = get_sphere('symmetric724')

    gtab = gradient_table(bvals, bvecs)

    from dipy.viz import fvtk

    evals = np.array([0.0015, 0.0003, 0.0003])
    evecs = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).T
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (90, 0)]
    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=[50, 50], snr=SNR)

    qb_odf_gt_single = single_tensor_qb_odf(sphere.vertices, evals, evecs)

    ren = fvtk.ren()

    response_actor = fvtk.sphere_funcs(qb_odf_gt_single, sphere)

    fvtk.add(ren, response_actor)

    fvtk.show(ren)

    ratio = sf_to_sh(qb_odf_gt_single, sphere, sh_order=8)

#    e1 = 15.0
#    e2 = 3.0
#    ratio = e2 / e1

    csd = ConstrainedSDTModel(gtab, ratio, None)

    csd_fit = csd.fit(S)

    ren = fvtk.ren()

    fodf = csd_fit.odf(sphere)

    response_actor = fvtk.sphere_funcs(fodf, sphere)

    fvtk.add(ren, response_actor)

    fvtk.show(ren)