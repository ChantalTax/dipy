""" Testing vizualization with fvtk
"""
import numpy as np

from dipy.viz import fvtk
from dipy import data

import numpy.testing as npt
import nose

@npt.dec.skipif(not fvtk.have_vtk)
@npt.dec.skipif(not fvtk.have_vtk_colors)
def test_fvtk_functions():

    # Create a renderer
    r = fvtk.ren()

    # Create 2 lines with 2 different colors
    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.random.rand(2, 3)
    c = fvtk.line(lines, colors)
    fvtk.add(r, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = fvtk.streamtube(lines, colors)
    c2.SetPosition(2, 0, 0)
    fvtk.add(r, c2)

    # Create a volume and return a volumetric actor using volumetric rendering
    vol = 100 * np.random.rand(100, 100, 100)
    vol = vol.astype('uint8')
    r = fvtk.ren()
    v = fvtk.volume(vol)
    fvtk.add(r, v)

    # Remove all objects
    fvtk.rm_all(r)

    # Put some text
    l = fvtk.label(r, text='Yes Men')
    fvtk.add(r, l)

    # Slice the volume
    fvtk.add(r, fvtk.slicer(vol, plane_i=[50]))

    # Change the position of the active camera
    fvtk.camera(r, pos=(0.6, 0, 0), verbose=False)

    fvtk.clear(r)

    # Peak directions
    p = fvtk.peaks(np.random.rand(3, 3, 3, 5, 3))
    fvtk.add(r, p)

    p2 = fvtk.peaks(np.random.rand(3, 3, 3, 5, 3),
                   np.random.rand(3, 3, 3, 5),
                   colors=(0, 1, 0))
    fvtk.add(r, p2)

    # Show everything
    # fvtk.show(r)


@npt.dec.skipif(not fvtk.have_vtk)
@npt.dec.skipif(not fvtk.have_vtk_colors)
def test_fvtk_ellipsoid():

    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)

    mevals = np.zeros((3, 2, 4, 3))
    mevecs = np.zeros((3, 2, 4, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')

    ren = fvtk.ren()

    fvtk.add(ren, fvtk.tensor(mevals, mevecs, sphere=sphere))

    fvtk.add(ren, fvtk.tensor(mevals, mevecs, np.ones(mevals.shape), sphere=sphere))

    npt.assert_equal(ren.GetActors().GetNumberOfItems(), 2)


def test_colormap():
    v = np.linspace(0., .5)
    map1 = fvtk.create_colormap(v, 'bone', auto=True)
    map2 = fvtk.create_colormap(v, 'bone', auto=False)
    npt.assert_(not np.allclose(map1, map2))

    npt.assert_raises(ValueError, fvtk.create_colormap, np.ones((2, 3)))
    npt.assert_raises(ValueError, fvtk.create_colormap, v, 'no such map')


def test_colormaps_matplotlib():
    # Skip this test if no matplotlib
    if not fvtk.have_matplotlib:
        raise npt.SkipTest

    v = np.random.random(1000)
    for name in 'jet', 'Blues', 'Accent', 'bone':
        # Matplotlib version of get_cmap
        rgba1 = fvtk.get_cmap(name)(v)
        # Dipy version of get_cmap
        rgba2 = data.get_cmap(name)(v)
        # dipy's colormaps are close to matplotlibs colormaps, but not perfect
        npt.assert_array_almost_equal(rgba1, rgba2, 1)

