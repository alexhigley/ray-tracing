# Updated to take into account April 2019 optic prescription changes.

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import astropy.units as u
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import curve_fit
import pdb
import sys
from copy import deepcopy
import textwrap

#sys.path.append('C:/python')
import PyXFocus.sources as sources
import PyXFocus.transformations as trans
import PyXFocus.surfaces as surfaces
import PyXFocus.analyses as analyses
import PyXFocus.conicsolve as conic
import OGRE.ogre_sim_alexplay as os

# Define final flight variables for OGRE mirror module.
z0 = 3500. * u.mm  # Z-coordinate of P-H intersection plane.
r_int = np.array([162.47158671, 165.00000000, 167.55028514, 170.11932985,
                  172.70727385, 175.31425796, 177.9404241, 180.58591531,
                  183.25087574, 185.9354507, 188.63978661, 191.36403106]) * u.mm
mirror_length = 100. * u.mm  # Axial length of mirror segments.
mirror_sep = 5. * u.mm  # Axial seperation of mirror segments.
thickness = 1. * u.mm  # Thickness of mirror segments in y-direction.

min_radial_extent = conic.primrad(z0 + mirror_sep/2, r_int[0], z0)
max_radial_extent = conic.primrad(z0 + mirror_length + mirror_sep/2,
                                  r_int[-1], z0)

# Find front/back radii of innermost optic.
zp_back = z0 + mirror_sep/2  # Axial position of parabola front.
zp_front = zp_back + mirror_length  # axial position of parabola back.
rp_front = u.Quantity([conic.primrad(zp_front, r, z0) for r in r_int])
rp_back = u.Quantity([conic.primrad(zp_back, r, z0) for r in r_int])

# Find radii of hyperbola.
zh_front = z0 - mirror_sep/2
zh_back = zh_front - mirror_length
rh_front = u.Quantity([conic.secrad(zh_front, r, z0) for r in r_int])
rh_back = u.Quantity([conic.secrad(zh_back, r, z0) for r in r_int])


def return_mirror_table():
    '''
    Returns table of relevant OGRE Mirror Module radii values,
    including the radii at the front/back of the primary mirrors,
    the radius at the intersection plane, and the radii at the
    front/back of the secondary mirrors.

    Note: Front radii are at a larger z-position (further from
    the focal plane).
    '''

    # Create table.
    mirror_table = Table(
        [np.arange(1, 13), rp_front, rp_back, r_int, rh_front, rh_back],
        names=['#', 'prim_front', 'prim_back', 'intersect', 'sec_front', 'sec_back'])

    # Format the radius values to four decimal places.
    mirror_table['prim_front'].format = '7.4f'
    mirror_table['prim_back'].format = '7.4f'
    mirror_table['intersect'].format = '7.4f'
    mirror_table['sec_front'].format = '7.4f'
    mirror_table['sec_back'].format = '7.4f'

    return mirror_table


# Define parameters for OGRE grating module.

# grating_length = 100.  # [mm] From old design.
# grating_thickness = 0.5  # [mm] From old design.
# l_hub_actual = 3248.923018287275  # From old design.
# l_hub_actual = 3248.8171629470453  # From old design.

grating_width = 70. * u.mm  # Updated design.
grating_length = 70. * u.mm  # Updated design.
grating_thickness = 0.5 * 70./100 * u.mm  # Updated design.
gammy = 1.5 * u.degree  # Incidence angle.
r_mid = np.median(r_int)  # Middle of optic. See OGRE #1 notebook.
groove_period = 6250. / u.mm
l_hub = 3250. * u.mm
l_hub_actual = 3248.942834145863 * u.mm  # Updated design.

# Groove period at center of grating.
groove_period = 1. / (1. / groove_period / (50.*u.mm + l_hub) * l_hub)

blaze = 30 * u.degree
yaw = np.arcsin(np.tan(blaze) * np.tan(gammy))
gamma = np.arcsin(np.sin(gammy) / np.cos(blaze))
blaze_wave = 2. / groove_period * np.sin(gamma) * np.sin(blaze)
fexvii_wave = 1.5014 * 3 * u.nm

convergence_length_mid = l_hub / np.cos(gammy)
z_mid = np.sqrt(convergence_length_mid**2 - r_mid**2)

# Define optimized grating positions.
# Blaze = 29.5 deg. From old design.
# zg_cen_right = np.array([3244.03873999, 3244.27660696, 3244.45496087,
#                          3244.7025949, 3244.90095772, 3245.13293622,
#                          3245.3800676, 3245.57579138, 3245.63637038,
#                          3245.69600153, 3245.7296269, 3245.81478748,
#                          3245.85691946, 3245.95352937, 3246.02333996,
#                          3246.08584086, 3246.14561958, 3246.1653864,
#                          3246.16614374])
#
# Blaze = 29.5 deg. From old design.
# zg_cen_left = np.array([3247.46397, 3247.47070629, 3247.36945324,
#                         3247.33221343, 3247.2843866, 3247.25381058,
#                         3247.21412667, 3247.1753826, 3247.13946169,
#                         3247.09864038, 3247.05528134, 3247.00144853,
#                         3246.95573546, 3246.95369672, 3246.94957945,
#                         3246.96661369, 3246.99890294, 3247.07513811,
#                         3247.104301])

# Old design, but still keeping it if we want to return to it.
if grating_length == 100*u.mm:
    # 1 mm intrafocal, blaze=30 deg
    zg_cen_right = np.array([3244.07536751, 3244.30026545, 3244.5080993,
                             3244.72224332, 3244.92324552, 3245.13824938,
                             3245.39050207, 3245.57873646, 3245.64175374,
                             3245.68251657, 3245.73333145, 3245.79588714,
                             3245.85485756, 3245.9199468, 3246.01291857,
                             3246.07960801, 3246.12706872, 3246.1464173,
                             3246.10459885]) * u.mm

    # 1 mm intrafocal, blaze=30 deg.
    zg_cen_left = np.array([3247.45944496, 3247.42478106, 3247.37476419,
                            3247.32576424, 3247.28080598, 3247.24513183,
                            3247.2181428, 3247.18147576, 3247.14209404,
                            3247.10470217, 3247.06488833, 3247.01356956,
                            3246.96543779, 3246.95875016, 3246.96154544,
                            3246.98431373, 3247.03062766, 3247.09252265,
                            3247.65522027]) * u.mm

    rg_cen = np.array([135.7903574126771, 138.4099036357215,
                       141.0293864267877, 143.64880238378686,
                       146.26814810507364, 148.8874201894639,
                       151.50661523625266, 154.1257298452312,
                       156.74476061670507, 159.36370415151157,
                       161.9825570510373, 164.60131591723592,
                       167.21997735264551, 169.83853796040637,
                       172.45699434427829, 175.07534310865864,
                       177.69358085859938, 180.31170419982482,
                       182.92970973874932]) * u.mm

    # Interpolate positions to account for grating thickness.
    left_interp = interp1d(rg_cen.value, zg_cen_left.value, kind='linear')
    right_interp = interp1d(rg_cen.value, zg_cen_right.value, kind='linear')

    # Account for grating thickness.
    rg_cen = rg_cen[0] + np.arange(16) * 3.117523493275094 * u.mm

    # 1 mm intrafocal, blaze = 30 deg.
    zg_cen_left = left_interp(rg_cen) * u.mm

    # 1 mm intrafocal, blaze = 30 deg.
    zg_cen_right = right_interp(rg_cen) * u.mm

# Updated design: 1 mm intrafocal, blaze = 30 deg.
elif grating_length == 70*u.mm:
    rg_cen, zg_cen_left, zg_cen_right = np.genfromtxt(
        '/Users/anh5866/Desktop/Coding/OGRE/PANTER_analysis/OGRE/Optimization/210324-Grating-Positions-1mm-Intrafocal-Fe-XVII.csv',
        skip_header=1, delimiter=',').transpose()
    zg_cen_left *= u.mm
    zg_cen_right *= u.mm
    rg_cen *= u.mm

    # Interpolate values so that we can account for grating thickness.
    left_interp = UnivariateSpline(rg_cen.value, zg_cen_left.value, k=4)
    right_interp = UnivariateSpline(rg_cen.value, zg_cen_right.value, k=4)

    # Account for grating thickness.
    rg_cen = rg_cen[0] + np.arange(24) * (
        np.sin(gammy) * grating_length + grating_thickness)

    # 1 mm intrafocal, blaze = 30 deg.
    zg_cen_left = left_interp(rg_cen) * u.mm

    # 1 mm intrafocal, blaze = 30 deg.
    zg_cen_right = right_interp(rg_cen) * u.mm

else:
    raise ValueError(textwrap.fill(textwrap.dedent('''
        grating_length is not a valid value. Please
        double-check the defined grating_length and ensure that
        it is defined properly.''')))

# Define number of gratings to reference later.
n_grat = len(rg_cen)


def beckmann_scatter(rays, h, rho, ripple, nind=1., lam=None):
    '''
    Add Beckmann scatter to a set of rays a la the SCATTER function
    in Webster Cash's IRT.

    This version is made faster by Bailey.
    '''

    # Define surface normal components.
    nx = rays[7]
    ny = rays[8]
    nz = rays[9]
    n = np.array([nx, ny, nz]).transpose()

    # Define direction cosines.
    qx = rays[4]
    qy = rays[5]
    qz = rays[6]
    q = np.array([qx, qy, qz]).transpose()

    # Compute some cross products.
    op = np.cross(n, q)
    norm = np.linalg.norm(op, axis=1)
    op[:,0] /= norm
    op[:,1] /= norm
    op[:,2] /= norm

    ip = np.cross(q, op)
    norm = np.linalg.norm(ip, axis=1)
    ip[:,0] /= norm
    ip[:,1] /= norm
    ip[:,2] /= norm

    dot = np.sum((q*n), axis=1)

    nel = len(rays[0])

    dq = np.zeros(np.shape(q))

    # Create *h* array.
    if type(h) is int or float:
        h = [h]
        nh = 1
    else:
        nh = len(list(h))

    h = np.array(h)

    # Create *rho* array.
    if type(rho) is int or float:
        rho = [rho]

    rho = np.array(rho)

    for ih in range(nh):
        if h[ih] <= 0.:
            continue
        hran = np.random.rand(nel)
        lamn = lam / nind
        gsc = 4. * np.pi * h[ih] * dot / lamn
        gsc = np.exp(-gsc * gsc)
        scat = [hran < gsc]

        grad = ((1. / np.random.rand(nel)) - 1.) * lamn/rho[ih]

        gthet = np.random.rand(nel) * 2 * np.pi
        dthet = grad * np.sin(gthet) / dot
        dphi = grad * np.cos(gthet)

        dq = dq + (dthet*ip + dphi*op) * scat

    ripr = np.random.randn(nel) * ripple
    rth = np.random.rand(nel) * 2 * np.pi
    dth = ripr * np.sin(rth)
    dph = ripr * np.cos(rth) * dot

    dq[:, 0] += (dth*ip[:, 0]) + (dph*op[:, 0])
    dq[:,1] += (dth*ip[:,1]) + (dph*op[:,1])
    dq[:,2] += (dth*ip[:,2]) + (dph*op[:,2])

    q = q + dq

    norm = np.linalg.norm(q, axis=1)
    q[:, 0] /= norm
    q[:, 1] /= norm
    q[:, 2] /= norm

    q = q.transpose()

    rays[4] = np.array(q[0], order='F')
    rays[5] = np.array(q[1], order='F')
    rays[6] = np.array(q[2], order='F')

    return


def beckmann_scatter_old(rays, h, rho, ripple, nind=1., lam=None):
    '''
    Add Beckmann scatter to a set of rays a la the SCATTER function
    in Webster Cash's IRT.
    '''

    # Define surface normal components.
    nx = rays[7]
    ny = rays[8]
    nz = rays[9]
    n = np.array([nx, ny, nz]).transpose()

    # Define direction cosines.
    qx = rays[4]
    qy = rays[5]
    qz = rays[6]
    q = np.array([qx, qy, qz]).transpose()

    # Compute some cross products.
    op = np.cross(n, q)
    op = np.array([(op[i] / np.linalg.norm(op[i])) for i in range(len(op))])

    ip = np.cross(q, op)
    ip = np.array([(ip[i] / np.linalg.norm(ip[i])) for i in range(len(ip))])

    dot = np.array([np.dot(a, b) for a, b in zip(q, n)])

    nel = len(rays[0])

    dq = np.zeros(np.shape(q))

    # Create *h* array.
    if type(h) is int or float:
        h = [h]
        nh = 1
    else:
        nh = len(list(h))

    h = np.array(h)

    # Create *rho* array.
    if type(rho) is int or float:
        rho = [rho]

    rho = np.array(rho)

    for ih in range(nh):
        if h[ih] <= 0.:
            continue
        hran = np.random.rand(nel)
        lamn = lam / nind
        gsc = 4. * np.pi * h[ih] * dot / lamn
        gsc = np.exp(-gsc * gsc)
        scat = [hran < gsc]

        grad = ((1. / np.random.rand(nel)) - 1.) * lamn/rho[ih]

        gthet = np.random.rand(nel) * 2 * np.pi
        dthet = grad * np.sin(gthet) / dot
        dphi = grad * np.cos(gthet)

        dq = dq + (dthet*ip + dphi*op) * scat

    ripr = np.random.randn(nel) * ripple
    rth = np.random.rand(nel) * 2 * np.pi
    dth = ripr * np.sin(rth)
    dph = ripr * np.cos(rth) * dot

    dq = dq + np.array([(dth[i]*ip[i] + dph[i]*op[i]) for i in range(nel)])
    q = q + dq
    q = np.array([(q[i] / np.linalg.norm(q[i])) for i in range(len(q))])
    q = q.transpose()

    rays[4] = np.array(q[0], order='F')
    rays[5] = np.array(q[1], order='F')
    rays[6] = np.array(q[2], order='F')

    return


def create_rays(n=100000, dphi=60*u.deg):
    '''
    Creates rays at intersection node of the OGRE mirror module
    covering the entire radial extent of the OGRE Mirror Module.

    Parameters
    ----------
    n : int
        Number of rays to create.
    dphi : Quantity
        Angular extent of rays to create.

    Returns
    -------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays at the intersection node of the OGRE mirror module.
    '''

    if type(dphi) != u.quantity.Quantity:
        raise ValueError(textwrap.fill(textwrap.dedent('''
            Angular extent argument (dphi) must be of the type
            astropy.units.quantity.Quantity. Please ensure that dphi
            has an astropy.unit associated with it.''')))

    # Convert number to integer if not already.
    n = int(n)

    # Define subannulus source.
    rays = sources.subannulus(min_radial_extent.value, max_radial_extent.value,
                              dphi.to('rad').value, n, zhat=1.)

    # Move to intersection node.
    trans.transform(rays, 0, 0, -z0.value, 0, 0, 0)

    # Rotate rays  so that dispersion axis is in the x-dimension.
    trans.transform(rays, 0, 0, 0, 0, 0, -np.pi/2)

    return rays


def ogre_mirror_module(rays, scatter=None, return_alpha=False, waves=None):
    '''
    Insert OGRE optic into optical system and trace rays through it.

    This function assumes the rays from the source have been
    generated and are at the intersection node of the optic.

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Ray object used in PyXFocus before propagating through
        OGRE mirror module.
    scatter : None or str
        Type of mirror scatter to simulate.
        Options: 'g' = Gaussian scatter
                 'b' = Beckmann scatter
                 'bg' = Beckmann + Gaussian scatter
                 None = No scatter
    return_alpha : boolean
        If true, will calculate the mean alpha angle of the
        paraboloid and hyperboloid mirrors. Default is False.
    waves : None or Quantity
        Array of wavelength values with the same length as number
        of rays [len(rays[0]) = len(waves)]. In units of nm.

    Returns
    -------
    alpha_p : numpy.ndarray
        Alpha values off of the primary (paraboloid) mirror.
    alpha_h : numpy.ndarray
        Alpha values off of the secondary (hyperboloid) mirror.

    OR

    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Ray object used in PyXFocus after propagating through
        OGRE mirror module.
    '''

    if waves is not None:
        if type(waves) != u.quantity.Quantity:
            raise ValueError(textwrap.fill(textwrap.dedent('''
                Wavelength(s) must be of the type
                astropy.units.quantity.Quantity. Please ensure
                that your wavelength value(s) have a valid
                astropy.units.quantity.Quantity associated
                with it.''')))

    if return_alpha is True:
        alpha_p = []
        alpha_h = []

    # Define blank PyXFocus ray object.
    new_rays = sources.annulus(0, 0, 0)

    # Loop through each mirror in mirror module.
    for i in range(len(r_int)):

        # Find photons which will hit this section of mirror.
        r = np.sqrt(rays[1]**2 + rays[2]**2)
        ind = np.where((r > rp_back[i].value) & (r < rp_front[i].value))[0]

        if len(ind) < 1:
            continue

        # Create new PyXFocus ray object with only the photons
        # that interact with this specific mirror.
        opd, x, y, z, l, m, n, ux, uy, uz = rays
        ind_rays = [opd[ind], x[ind], y[ind], z[ind], l[ind],
                    m[ind], n[ind], ux[ind], uy[ind], uz[ind]]

        # Propagate these photons to the paraboloid.
        surfaces.wolterprimary(ind_rays, r_int[i].value, z0.value)

        # Find which photons will interact with paraboloid.
        ind = np.where(
            (ind_rays[3] > (z0 + mirror_sep/2).value) &
            (ind_rays[3] < (z0 + mirror_sep/2 + mirror_length).value))

        # Keep only the photons which interact with the actual size
        # of the mirror.
        opd, x, y, z, l, m, n, ux, uy, uz = ind_rays
        ind_rays = [opd[ind], x[ind], y[ind], z[ind], l[ind],
                    m[ind], n[ind], ux[ind], uy[ind], uz[ind]]

        # Calculate alpha if desired.
        if return_alpha is True:
            alpha_p.append(np.mean(analyses.grazeAngle(ind_rays)))

        # Reflect photons off of primary.
        trans.reflect(ind_rays)

        # Add Beckmann scatter if desired.
        if scatter == 'b':
            beckmann_scatter(ind_rays, 0, 0, 1.48e-5)

        # Add Beckmann + Gaussian scatter if desired. Values
        # produce ~1" FWHM on OGRE focal plane.
        if scatter == 'bg':
            beckmann_scatter(ind_rays, 0, 0, 1.3e-5)
            # beckmann_scatter(ind_rays, 0, 0, 9e-6)
            # ind_rays[4] += np.random.normal(scale=6e-7, size=len(ind_rays[4]))
            # ind_rays[5] += np.random.normal(scale=1.2e-6, size=len(ind_rays[5]))
            # ind_rays[6] = np.sqrt(1. - ind_rays[5]**2 - ind_rays[4]**2)

        # Propagate photons to the secondary mirror.
        surfaces.woltersecondary(ind_rays, r_int[i].value, z0.value)

        # Find which photons will interact with hyperboloid.
        ind = np.where(
            (ind_rays[3] < (z0 - mirror_sep/2).value) &
            (ind_rays[3] > (z0 - mirror_sep/2 - mirror_length).value))

        # Keep only the photons which will interact with this mirror
        # and create new PyXFocus ray object to store these photons.
        opd, x, y, z, l, m, n, ux, uy, uz = ind_rays
        ind_rays = [opd[ind], x[ind], y[ind], z[ind], l[ind],
                    m[ind], n[ind], ux[ind], uy[ind], uz[ind]]

        # Calculate alpha_h if desired.
        if return_alpha is True:
            alpha_h.append(np.mean(analyses.grazeAngle(ind_rays)))

        # Reflect the photons off of the secondary.
        trans.reflect(ind_rays)

        # Add Gaussian scatter if desired.
        if scatter == 'g':
            ind_rays[4] += np.random.normal(scale=2.e-6, size=len(ind_rays[4]))
            ind_rays[5] += np.random.normal(scale=1.5e-5, size=len(ind_rays[5]))
            ind_rays[6] = np.sqrt(1. - ind_rays[5]**2 - ind_rays[4]**2)

        if scatter == 'bg':
            ind_rays[4] += np.random.normal(scale=5e-7, size=len(ind_rays[4]))
            # ind_rays[5] += np.random.normal(scale=0.9e-5, size=len(ind_rays[5]))
            ind_rays[6] = np.sqrt(1. - ind_rays[5]**2 - ind_rays[4]**2)

        # Add to 'master' PyXFocus ray object.
        new_rays = [np.append(new_rays[i], ind_rays[i])
                    for i in range(len(new_rays))]

    # Copy rays.
    rays = trans.copy_rays(new_rays)

    # Calculate mean alphas if desired.
    if return_alpha is True:
        print('Mean alpha_p: ' + str(np.mean(alpha_p)))
        print('Mean alpha_h: ' + str(np.mean(alpha_h)))
        return alpha_p, alpha_h

    # Return PyXFocus ray object.
    else:
        return rays


def create_g_misalign(std_pitch, std_roll, std_yaw, std_x, std_y, std_z):
    """
    Creates an array of individual grating misalignments for use in the
    ogre_grating_module function.

    This function assumes a normal distribution of misalignments,
    centered around zero and with specified standard deviations.

    Parameters
    ----------
    std_pitch : Quantity or float
        Standard deviation of individual grating pitch in left/right
        grating stacks.
    std_roll : Quantity or float
        Standard deviation of individual grating roll in left/right
        grating stacks.
    std_yaw : Quantity or float
        Standard deviation of individual grating yaw in left/right
        grating stacks.
    std_x : Quantity or float
        Standard deviation of individual grating x-positions in
        left/right grating stacks.
    std_y : Quantity or float
        Standard deviation of individual grating y-positions in
        left/right grating stacks.
    std_z : Quantity or float
        Standard deviation of individual grating z-positions in
        left/right grating stacks.

    Returns
    -------
    g_misalign : list
        Misalignment list of astropy.unit.quantity.Quantity to pass
        to ogre_grating_module.

    """

    # Create lists of passed arguments and their variable names.
    arg_names = ['std_pitch', 'std_roll', 'std_yaw', 'std_x', 'std_y', 'std_z']
    arg = [std_pitch, std_roll, std_yaw, std_x, std_y, std_z]

    # Create dictionary to make parsing easier.
    d = {n:a for n,a in zip(arg_names, arg)}

    # Adopt units if they are not provided.
    for dof in ['std_pitch', 'std_roll', 'std_yaw']:
        if type(d[dof]) != u.quantity.Quantity:
            print('Unit for ' + dof + ' not specified. Adopting [rad]...')
            d[dof] *= u.rad

    for dof in ['std_x', 'std_y', 'std_z']:
        if type(d[dof]) != u.quantity.Quantity:
            print('Unit for ' + dof + ' not specified. Adopting [mm]...')
            d[dof] *= u.mm

    g_misalign = []
    for key in arg_names:
        mis_values = np.random.normal(0, d[key].value, n_grat*2) * d[key].unit
        g_misalign.append(mis_values)

    return g_misalign


def create_s_misalign(pitch, roll, yaw, x, y, z):
    '''
    Creates a dictionary of misalignments for the left/right grating
    stacks to use with the ogre_grating_module() function.

    Assumes half of the misalignment is attributed to the left grating
    stack and the other half is attributed to the right grating stack in
    the opposite direction.

    Parameters
    ----------
    pitch : Quantity or float
        Standard deviation of individual grating pitch in left/right
        grating stacks.
    roll : Quantity or float
        Standard deviation of individual grating roll in left/right
        grating stacks.
    yaw : Quantity or float
        Standard deviation of individual grating yaw in left/right
        grating stacks.
    x : Quantity or float
        Standard deviation of individual grating x-positions in
        left/right grating stacks.
    y : Quantity or float
        Standard deviation of individual grating y-positions in
        left/right grating stacks.
    z : Quantity or float
        Standard deviation of individual grating z-positions in
        left/right grating stacks.

    Returns
    -------
    s_misalign : dict
        Misalignment dictionary with keys "lgrat" and "rgrat" that
        contain lists of astropy.unit.quantity.Quantity to pass to
        ogre_grating_module().
    '''

    # Create lists of passed arguments and their variable names.
    arg_names = ['pitch', 'roll', 'yaw', 'x', 'y', 'z']
    arg = [pitch, roll, yaw, x, y, z]

    # Create dictionary to make parsing easier.
    d = {n:a for n,a in zip(arg_names, arg)}

    # Adopt units if they are not provided.
    for dof in ['pitch', 'roll', 'yaw']:
        if type(d[dof]) != u.quantity.Quantity:
            print('Unit for ' + dof + ' not specified. Adopting [rad]...')
            d[dof] *= u.rad

    for dof in ['x', 'y', 'z']:
        if type(d[dof]) != u.quantity.Quantity:
            print('Unit for ' + dof + ' not specified. Adopting [mm]...')
            d[dof] *= u.mm

    # Create dictionary to be returned.
    s_misalign = {}
    s_misalign['lgrat'] = [d['pitch']/2, d['roll']/2, d['yaw']/2,
                           d['x']/2, d['y']/2, d['z']/2]
    s_misalign['rgrat'] = [-d['pitch']/2, -d['roll']/2, -d['yaw']/2,
                           -d['x']/2, -d['y']/2, -d['z']/2]

    return s_misalign


# Define grating stack & module parameters.
cen_width = 5 * u.mm
edge_width = 5 * u.mm
mod_width = 150. * u.mm
grating_width = mod_width/2 - cen_width/2 - edge_width


def ogre_grating_module(rays, waves=blaze_wave, yaw=yaw, g_misalign=None,
                        s_misalign=None, m_misalign=None, plot=False,
                        inverted=False, return_diff_ind=False):
    '''
    Insert OGRE grating module into optical system and trace rays.

    This function assumes the rays having already passed through the
    OGRE optic.

    All misalignments are in the form [pitch, roll, yaw, x, y, z].

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Ray object used in PyXFocus before propagating through
        OGRE Grating Module.
    waves : Quantity
        Wavelength of photons corresponding to rays object.
    yaw : Quantity
        Yaw of grating module. Derived from Littrow configuration
        for blaze wavelength.
    g_misalign : None or list
        Misalignments of individual gratings in left/right grating
        stacks. Create these misalignments with the create_g_misalign()
        function!
    s_misalign : None or dict
        Misalignments of left/right stacks relative to nominal
        orientation. Created with the create_s_misalign() function!
    m_misalign : None or list
        Mislaignments of module relative to nominal orientation.
    plot : boolean
        Determines whether to plot rays when propagating through
        the grating module.
    inverted : boolean
        Determines whether to invert the grating module function. This
        will allow modeling of the mirrored grating module.
    return_diff_ind : boolean
        If True, the function will return the indicies of the initially
        passed ray object that were diffracted.
    '''

    # Find mean z-hat location of gratings.
    z_mean = (np.mean(zg_cen_left) + np.mean(zg_cen_right))/2

    # Create zero misalignments if no module misalignments are specified.
    if m_misalign is None:
        m_misalign = np.zeros(6)

    # Misalign entire grating module (two stacks).
    m_dpitch, m_droll, m_dyaw, m_dx, m_dy, m_dz = m_misalign
    trans.transform(rays, 0, rg_cen.mean().to('mm').value,
                    z_mean.to('mm').value, 0, 0, 0)
    trans.transform(rays, m_dx, m_dy, m_dz, m_dpitch, m_dyaw, m_droll)
    trans.itransform(rays, 0, rg_cen.mean().to('mm').value,
                     z_mean.to('mm').value, 0, 0, 0)

    # Divide rays in two halfs: "left" stack (+x) and "right" stack (-x)
    l_ind = np.where(
        (rays[1] > (0. + cen_width.to('mm').value/2)) &
        (rays[1] < (grating_width + cen_width/2).to('mm').value))[0]
    lrays = [r[l_ind] for r in rays]

    r_ind = np.where(
        (rays[1] < (0. - cen_width.to('mm').value/2)) &
        (rays[1] > -(grating_width + cen_width/2).to('mm').value))[0]
    rrays = [r[r_ind] for r in rays]

    # Plot rays if desired.
    if plot is True:
        # Plot divided rays.
        plt.ion()
        plt.figure()
        plt.scatter(lrays[1], lrays[2], s=0.5, label='Left Rays')
        plt.scatter(rrays[1], rrays[2], s=0.5, label='Right Rays')
        plt.axis('equal')
        plt.xlabel('X [mm]')
        plt.ylabel('Y [mm]')

    if type(waves) == u.quantity.Quantity:
        if type(waves.value) is int or type(waves.value) is float:
            waves = waves * np.ones(len(rays[0]))
        elif len(waves) == len(rays[0]):
            pass
        else:
            raise ValueError(textwrap.fill(textwrap.dedent('''
                Wavelength list/array must have the same length as each array
                in the rays variable. Please ensure these two arrays have the
                same length.''')))
    else:
        raise ValueError(textwrap.fill(textwrap.dedent('''
            Wavelength variable (waves) must be either a int/float with
            associated astropy.units.quantity.Quantity or an array of
            astropy.units.quantity.Quantity.''')))

    # Split up individual grating misalignments to left/right stacks.
    if g_misalign is not None:
        if np.shape(g_misalign) != (6, n_grat * 2):
            raise ValueError(textwrap.fill(textwrap.dedent('''
                g_misalign does not have the correct shape. Please ensure that
                the g_misalign variable has the correct shape. You can use the
                create_g_misalign() function to create this variable.''')))

        l_grat_misalign = [misalign[:n_grat] for misalign in g_misalign]
        r_grat_misalign = [misalign[n_grat:] for misalign in g_misalign]

    # Split stack misalignments.
    if s_misalign is None:
        s_misalign = {}
        s_misalign['lgrat'] = [0*u.rad, 0*u.rad, 0*u.rad, 0*u.mm, 0*u.mm, 0*u.mm]
        s_misalign['rgrat'] = [0*u.rad, 0*u.rad, 0*u.rad, 0*u.mm, 0*u.mm, 0*u.mm]

    # Propagate rays through lrays through left grating array.
    rays = trans.copy_rays(lrays)
    diff_inds = np.array([])

    # Copy grating z-positions.
    zg_cen = deepcopy(zg_cen_left)

    # Misalign coordinate system to simulate gross module misalignments.
    dpitch, droll, dyaw, dx, dy, dz = s_misalign['lgrat']

    trans.transform(rays, (grating_width/2 + cen_width/2).to('mm').value,
                    rg_cen.to('mm').value.mean(), zg_cen.to('mm').value.mean(),
                    0, 0, 0)
    trans.transform(
        rays, dx.to('mm').value, dy.to('mm').value, dz.to('mm').value,
        dpitch.to('rad').value, dyaw.to('rad').value, droll.to('rad').value)
    trans.itransform(rays, (grating_width/2 + cen_width/2).to('mm').value,
                    rg_cen.to('mm').value.mean(), zg_cen.to('mm').value.mean(),
                    0, 0, 0)

    # Step through left grating stack.
    for i in range(n_grat):

        # Define coordinate system so that we can return to it later.
        glob_coords = [trans.tr.identity_matrix()] * 4

        # Move to grating location.
        trans.transform(rays, (grating_width/2 + cen_width/2).to('mm').value,
                        rg_cen[i].to('mm').value, zg_cen[i].to('mm').value,
                        0, 0, 0, coords=glob_coords)

        # Rotate to angle of beam.
        trans.transform(rays, 0, 0, 0, -np.pi/2, 0, 0, coords=glob_coords)
        trans.transform(
            rays, 0, 0, 0, -np.arctan((rg_cen[i]/zg_cen[i]).to('').value),
            0, 0, coords=glob_coords)

        # Put incidence angle onto grating.
        trans.transform(
            rays, 0, 0, 0, gammy.to('rad').value, 0, 0, coords=glob_coords)

        # Get +y pointing towards grating surface for radgrat function.
        trans.transform(rays, 0, 0, 0, 0, 0, np.pi, coords=glob_coords)

        # Add yaw.
        yaw_add = np.arctan(
            ((grating_width/2 + cen_width/2) / zg_cen[i]).to('').value)
        trans.transform(rays, 0, 0, 0, 0, 0,
                        -yaw.to('rad').value + yaw_add, coords=glob_coords)

        # Add grating misalignments.
        if g_misalign is not None:
            l_pitch = l_grat_misalign[0][i].to('rad').value
            l_roll = l_grat_misalign[1][i].to('rad').value
            l_yaw = l_grat_misalign[2][i].to('rad').value
            l_x = l_grat_misalign[3][i].to('mm').value
            l_y = l_grat_misalign[4][i].to('mm').value
            l_z = l_grat_misalign[5][i].to('mm').value
            trans.transform(
                rays, l_x, l_z, l_y, l_pitch, l_roll, l_yaw, coords=glob_coords)

        # Go to hub location.
        trans.transform(
            rays, 0, -l_hub.to('mm').value, 0, 0, 0, 0,  coords=glob_coords)

        # Project photons onto xy plane (grating surface).
        surfaces.flat(rays)

        # Find which photons will interact with the grating surface.
        ray_ind = np.where(
            (rays[2] > (l_hub - grating_length/2).to('mm').value) &
            (rays[2] < (l_hub + grating_length/2).to('mm').value) &
            (np.abs(rays[1]) < grating_width.to('mm').value/2))[0]

        # Remove indices which have already been reflected/diffracted.
        ray_ind = ray_ind[np.isin(ray_ind, diff_inds, invert=True)]
        ray_ind = np.array(ray_ind, dtype=int)
        diff_inds = np.concatenate((ray_ind, diff_inds))

        # If there are no rays that interact with the grating, continue.
        if len(ray_ind) < 1:
            rays = trans.applyT(rays, glob_coords, inverse=True)
            continue

        # Reflect photons which fall onto this grating.
        trans.reflect(rays, ind=ray_ind)

        # Diffract photons.
        trans.radgrat(
            rays, (1e6/groove_period/l_hub).to('').value, 1,
            waves[l_ind].to('nm', equivalencies=u.spectral()).value,
            ind=ray_ind)

        # Return back to hub coordinate system.
        rays = trans.applyT(rays, glob_coords, inverse=True)

    diff_inds = np.array(diff_inds, dtype=int)
    rays = [r[diff_inds] for r in rays]

    # Realign stack coordinate system.
    trans.transform(rays, (grating_width/2 + cen_width/2).to('mm').value,
                    rg_cen.to('mm').value.mean(), zg_cen.to('mm').value.mean(),
                    0, 0, 0)
    trans.itransform(
        rays, dx.to('mm').value, dy.to('mm').value, dz.to('mm').value,
        dpitch.to('rad').value, dyaw.to('rad').value, droll.to('rad').value)
    trans.itransform(rays, (grating_width/2 + cen_width/2).to('mm').value,
                     rg_cen.to('mm').value.mean(), zg_cen.to('mm').value.mean(),
                     0, 0, 0)

    lrays = trans.copy_rays(rays)
    l_diff_ind = deepcopy(l_ind[diff_inds])

    # Onto right grating array, which is very similar to the left array.
    rays = trans.copy_rays(rrays)
    diff_inds = np.array([])

    # Redefine z-position variable.
    zg_cen = deepcopy(zg_cen_right)

    # Generate misalignments.
    dpitch, droll, dyaw, dx, dy, dz = s_misalign['rgrat']

    # Misalign coordinate system to simulate gross module misalignments.
    trans.transform(rays, -(grating_width/2 + cen_width/2).to('mm').value,
                    rg_cen.mean().to('mm').value,
                    zg_cen.mean().to('mm').value, 0, 0, 0)
    trans.transform(
        rays, dx.to('mm').value, dy.to('mm').value, dz.to('mm').value,
        dpitch.to('rad').value, dyaw.to('rad').value, droll.to('rad').value)
    trans.itransform(rays, -(grating_width/2 + cen_width/2).to('mm').value,
                     rg_cen.mean().to('mm').value,
                     zg_cen.mean().to('mm').value, 0, 0, 0)

    for i in range(n_grat):

        # Define coordinate system to return to later.
        glob_coords = [trans.tr.identity_matrix()] * 4

        # Move to grating location.
        trans.transform(rays, -(grating_width/2 + cen_width/2).to('mm').value,
                        rg_cen[i].to('mm').value, zg_cen[i].to('mm').value,
                        0, 0, 0, coords=glob_coords)

        # Rotate to angle of beam.
        trans.transform(rays, 0, 0, 0, -np.pi/2, 0, 0, coords=glob_coords)
        trans.transform(
            rays, 0, 0, 0, -np.arctan((rg_cen[i] / zg_cen[i]).to('').value),
            0, 0, coords=glob_coords)

        # Put incidence angle onto grating.
        trans.transform(
            rays, 0, 0, 0, gammy.to('rad').value, 0, 0, coords=glob_coords)

        # Get +y pointing towards grating surface for radgrat function.
        trans.transform(rays, 0, 0, 0, 0, 0, np.pi, coords=glob_coords)

        # Add yaw.
        yaw_add = np.arctan(
            ((grating_width + cen_width) / 2 / zg_cen[i]).to('').value)
        trans.transform(
            rays, 0, 0, 0, 0, 0, -yaw.to('rad').value - yaw_add,
            coords=glob_coords)

        # Add grating misalignments.
        if g_misalign is not None:
            r_pitch = r_grat_misalign[0][i].to('rad').value
            r_roll = r_grat_misalign[1][i].to('rad').value
            r_yaw = r_grat_misalign[2][i].to('rad').value
            r_x = r_grat_misalign[3][i].to('mm').value
            r_y = r_grat_misalign[4][i].to('mm').value
            r_z = r_grat_misalign[5][i].to('mm').value
            trans.transform(
                rays, r_x, r_z, r_y, r_pitch, r_roll, r_yaw, coords=glob_coords)

        # Go to hub location.
        trans.transform(
            rays, 0, -l_hub.to('mm').value, 0, 0, 0, 0,  coords=glob_coords)

        # Project photons onto xy plane (grating surface).
        surfaces.flat(rays)

        # Find which photons will interact with the grating surface.
        ray_ind = np.where(
            (rays[2] > (l_hub - mirror_length/2).to('mm').value) &
            (rays[2] < (l_hub + mirror_length/2).to('mm').value) &
            (np.abs(rays[1]) < grating_width.to('mm').value/2))[0]

        # Remove indices which have already been reflected/diffracted.
        ray_ind = ray_ind[np.isin(ray_ind, diff_inds, invert=True)]
        ray_ind = np.array(ray_ind, dtype=int)
        diff_inds = np.concatenate((ray_ind, diff_inds))

        # If there are no rays that interact with the grating, continue.
        if len(ray_ind) < 1:
            rays = trans.applyT(rays, glob_coords, inverse=True)
            continue

        # Reflect photons which fall onto this grating.
        trans.reflect(rays, ind=ray_ind)

        # Diffract photons.
        trans.radgrat(
            rays, (1e6/groove_period/l_hub).to('').value, 1,
            waves[r_ind].to('nm', equivalencies=u.spectral()).value,
            ind=ray_ind)

        # Return back to hub coordinate system.
        rays = trans.applyT(rays, glob_coords, inverse=True)

    diff_inds = np.array(diff_inds, dtype=int)
    rays = [r[diff_inds] for r in rays]

    # Realign stack coordinate system.
    trans.transform(rays, -(grating_width/2 + cen_width/2).to('mm').value,
                    rg_cen.mean().to('mm').value, zg_cen.mean().to('mm').value,
                    0, 0, 0)
    trans.itransform(
        rays, dx.to('mm').value, dy.to('mm').value, dz.to('mm').value,
        dpitch.to('rad').value, dyaw.to('rad').value, droll.to('rad').value)
    trans.itransform(rays, -(grating_width/2 + cen_width/2).to('mm').value,
                    rg_cen.mean().to('mm').value, zg_cen.mean().to('mm').value,
                    0, 0, 0)

    rrays = trans.copy_rays(rays)
    r_diff_ind = deepcopy(r_ind[diff_inds])

    # Combine lrays & rrays into single ray object.
    rays = list(np.concatenate((lrays, rrays), axis=1))
    diff_ind = np.concatenate((l_diff_ind, r_diff_ind))

    # Realign module coordinate system.
    trans.transform(rays, 0, rg_cen.mean().to('mm').value,
                    z_mean.to('mm').value, 0, 0, 0)
    trans.itransform(rays, m_dx, m_dy, m_dz, m_dpitch, m_dyaw, m_droll)
    trans.itransform(rays, 0, rg_cen.mean().to('mm').value,
                    z_mean.to('mm').value, 0, 0, 0)

    if return_diff_ind:
        return rays, diff_ind
    else:
        return rays


def grating_rib_occultation(rays, waves=None, orders=None, rib_width=1*u.mm,
                            rib_num=5, cen_width=cen_width,
                            grating_width=grating_width):
    '''
    Occults rays passing through grating array (ribs, stack alignment,
    support structure, etc.)

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays to occult.
    waves : None or Quantity
        Wavelength values corresponding to rays to occult (if provided).
    orders : None or numpy.ndarray
        Diffraction order values to occult that correspond to
        wavelengths and rays provided.
    rib_width : Quantity
        Width of individual rib.
    rib_num : int
        Number of ribs on each grating.

    Returns
    -------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Occulted rays.

    If waves & orders are not None:

    waves : Quantity
        Occulted wavelengths.
    orders : numpy.ndarray
        Occulted diffraction orders.
    '''

    # Value error if rib_width and rib_num are not both specified.
    if rib_width != None and rib_num == None:
        raise ValueError(textwrap.fill(textwrap.dedent('''
            You have specified a rib width, but not a valid number of ribs.
            Please specify a valid number of ribs.''')))

    if rib_width == None and rib_num != None:
        raise ValueError(textwrap.fill(textwrap.dedent('''
            You have specified the number of ribs, but not a valid rib width.
            Please specify a valid rib width.''')))

    # Find mean z-hat location of gratings.
    z_mean = (np.mean(zg_cen_left) + np.mean(zg_cen_right))/2

    # Move them to the mean grating center point.
    trans.transform(rays, 0, 0, z_mean.to('mm').value, 0, 0, 0)
    surfaces.flat(rays)
    rays[3] = np.full(len(rays[3]), z_mean.to('mm').value)

    # Find rib positions.
    rib_cen_positions = np.linspace(
        rib_width.to('mm').value/2,
        (grating_width - rib_width/2).to('mm').value, rib_num) * u.mm + cen_width/2
    rib_ind = np.array([], dtype=int)
    for rc in rib_cen_positions:
        ri = np.where((np.abs(rays[1]) > (rc - rib_width/2).to('mm').value)
                      & (np.abs(rays[1]) < (rc + rib_width/2).to('mm').value))[0]
        rib_ind = np.concatenate((rib_ind, ri))

    # Remove occulted rays.
    rays = [np.delete(r, rib_ind) for r in rays]

    if waves or orders is not None:
        return rays, np.delete(waves, rib_ind), np.delete(orders, rib_ind)
    else:
        return rays


def spectrometer_focal_plane(rays, yaw=yaw, inverted=False):
    '''
    Transforms rays to the nominal spectrometer focal plane for a single
    60 degree azimuthal span.

    Note: This is not the OGRE focal plane, as each detector will
    share two diffraction arcs.

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays after passing through grating array.
    yaw : Quantity
        Yaw of grating module.
    inverted : boolean
        Determines whether or not we're dealing with an inverted grating
        module, or one that diffracts in the opposite direction.

    Returns
    -------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays after being propagated onto spectrometer focal plane.
    '''

    if inverted is True:
        yaw = -yaw

    # Move to 'center' grating in module and rotate into grating frame.
    trans.transform(rays, 0, r_mid, z_mid, 0, 0, 0)
    trans.transform(rays, 0, 0, 0, -np.arctan(r_mid/z_mid) + gammy, 0, 0)
    trans.transform(rays, 0, 0, 0, 0, -yaw, 0)
    trans.transform(rays, 0, 0, 0, 0, np.pi, 0)

    # Move to spectrometer focal plane.
    trans.transform(rays, 0, 0, l_hub_actual, 0, 0, 0)
    surfaces.flat(rays)

    return rays


# Define detector location variables.
spect_focus = -2.870 * u.mm
det_x, det_y = 98.246 * u.mm, 170.168 * u.mm

# Define detector variable.s
pixel_size = 16.e-3
n_pix_x = 1632
n_pix_y = 1608


def ogre_focal_plane(rays, fp_misalignments=[0, 0, 0, 0, 0, 0],
                     det_misalignments=[0, 0, 0, 0, 0, 0]):
    '''
    Transforms rays to the nominal spectrometer focal plane.

    Note: This is the OGRE focal plane, with each detector
    sharing two diffraction arcs.

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays to be propagated to OGRE focal plane.
    fp_misalignments : list or numpy.ndarray
        Misalignments of entire focal plane.
    det_misalignments : list or numpy.ndarray
        Misalignments of individual detector on focal plane.

    Returns
    -------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays after propagating to OGRE focal plane.
    '''

    # Move to the focal plane of the OGRE spectrograph.
    # spect_focus = -2.6853525103924767
    trans.transform(rays, 0, 0, spect_focus.to('mm').value, 0, 0, 0)
    surfaces.flat(rays)

    # Add gross focal plane misalignments.
    dpitch, droll, dyaw, dx, dy, dz = fp_misalignments
    trans.transform(rays, dx, dy, dz, dpitch, dyaw, droll)
    surfaces.flat(rays)

    # Grab detector misalignments.
    dpitch, droll, dyaw, dx, dy, dz = det_misalignments

    # Move to detector location and misalign detector.
    glob_coords = [trans.tr.identity_matrix()] * 4
    trans.transform(
        rays, det_x.to('mm').value, det_y.to('mm').value,
        0, 0, 0, 0, coords=glob_coords)
    trans.transform(rays, 0, 0, 0, 0, 0, -np.radians(30.), coords=glob_coords)
    trans.transform(rays, dx, dy, dz, dpitch, dyaw, droll)
    surfaces.flat(rays)

    # Return to starting coordinate system.
    rays = trans.applyT(rays, glob_coords, inverse=True)

    return rays


def create_histogram_of_detector(rays, pixel_size=pixel_size):
    '''
    Bins rays on the focal plane into detector pixels.

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays on OGRE focal plane.
    pixel_size : float
        Pixel size of OGRE detector.

    Returns
    -------
    H : numpy.ndarray
        Binned 2D array of counts in detector pixels.
    xedges : numpy.ndarray
        1D array of x-edges for 2D histogram.
    yedges : numpy.ndarray
        1D array of y-edges for 2D histogram.
    '''

    # Copy rays.
    c_rays = deepcopy(rays)

    # Move to detector center.
    glob_coords = [trans.tr.identity_matrix()] * 4
    trans.transform(
        c_rays, det_x.to('mm').value, det_y.to('mm').value,
        0, 0, 0, 0, coords=glob_coords)
    trans.transform(
        c_rays, 0, 0, 0, 0, 0, -np.radians(30.), coords=glob_coords)

    # Find X & Y edges.
    bin_edges_x = np.arange(-n_pix_x/2, n_pix_x/2 + 1) * pixel_size
    bin_edges_y = np.arange(-n_pix_y/2, n_pix_y/2 + 1) * pixel_size

    # Keep only photons which fall onto detector.
    ind = np.where((c_rays[1] > bin_edges_x[0]) &
                   (c_rays[1] < bin_edges_x[-1]) &
                   (c_rays[2] > bin_edges_y[0]) &
                   (c_rays[2] < bin_edges_y[-1]))

    # Bin rays.
    H, xedges, yedges = np.histogram2d(
        c_rays[1][ind], c_rays[2][ind], bins=(bin_edges_x, bin_edges_y))
    H = H.T

    # Return to starting coordinate system.
    trans.applyT(c_rays, glob_coords, inverse=True)

    return H, xedges, yedges


def bin_rays(rays, waves, orders, pixel_size=pixel_size):
    '''
    Bins rays into detector pixels and creates ray_table which gives
    position on detector in physical space and detector pixels.

    Note: This function is used to create variable to pass to the
    create_spectrum() function.

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays on the OGRE focal plane to be binned.
    waves : Quantity
        Wavelength values corresponding to each ray in rays.
    orders : numpy.ndarray
        Diffraction orders corresponding to each ray in rays.
    pixel_size : float
        Pixel size of OGRE detector pixels.

    Returns
    -------
    ray_table : astropy.table
        Table describing counts on the detector.
    '''
    # Copy rays.
    c_rays = deepcopy(rays)

    # Amend

    # Move to detector center.
    glob_coords = [trans.tr.identity_matrix()] * 4
    trans.transform(
        c_rays, det_x.to('mm').value + 0, det_y.to('mm').value,
        0, 0, 0, 0, coords=glob_coords)
    trans.transform(
        c_rays, 0, 0, 0, 0, 0, -np.radians(30.), coords=glob_coords)

    # Find X & Y edges.
    bin_edges_x = np.arange(-n_pix_x/2, n_pix_x/2 + 1) * pixel_size
    bin_edges_y = np.arange(-n_pix_y/2, n_pix_y/2 + 1) * pixel_size

    # Create Table to store ray x,y positions in.
    ray_table = Table([c_rays[1], c_rays[2], waves, orders],
                      names=['x', 'y', 'wave', 'order'])

    # Find x,y coordinates on detector for each ray.
    ref_val = 100000
    x_pix = Column(np.full(len(c_rays[1]), ref_val, dtype=float), name='x_pix')
    y_pix = Column(np.full(len(c_rays[1]), ref_val, dtype=float), name='y_pix')

    for i in range(len(bin_edges_x) - 1):
        ind = np.where((ray_table['x'] > bin_edges_x[i]) &
                       (ray_table['x'] < bin_edges_x[i+1]))[0]
        x_pix[ind] = bin_edges_x[i] + pixel_size/2

    for i in range(len(bin_edges_y) - 1):
        ind = np.where((ray_table['y'] > bin_edges_y[i]) &
                       (ray_table['y'] < bin_edges_y[i+1]))[0]
        y_pix[ind] = bin_edges_y[i] + pixel_size/2

    # Append columns to ray_table.
    ray_table.add_columns([x_pix, y_pix])

    remove_ind = np.where(
        (ray_table['x_pix'] == ref_val) | (ray_table['y_pix'] == ref_val))

    # Remove rows that do not fall on detector.
    ray_table.remove_rows(remove_ind)

    # Return to starting coordinate system.
    trans.applyT(rays, glob_coords, inverse=True)

    return ray_table


def extract_counts_from_binned_rays(H, xedges, yedges):
    '''
    Extract counts from binned rays produced by the
    create_histogram_on_detector() function.
    '''

    # Define x/y pixel centers.
    xcen = xedges[:-1] + np.diff(xedges)/2
    ycen = yedges[:-1] + np.diff(yedges)/2

    # Find inds where H > 0.
    yinds, xinds = np.where(H > 0)

    # Create x,y count positions.
    xcounts, ycounts = np.array([]), np.array([])
    for yind, xind in zip(yinds, xinds):
        # Find how many counts in pixel.
        n_counts = int(H[yind, xind])
        # Add counts to list.
        x, y = np.ones(n_counts) * xcen[xind], np.ones(n_counts) * ycen[yind]
        xcounts = np.concatenate([xcounts, x])
        ycounts = np.concatenate([ycounts, y])

    return xcounts, ycounts


def points_to_dispersion_dir(x, y):
    '''
    Rotates counts on detector to global dispersion / cross-dispersion
    coordinate system.

    Parameters
    ----------
    x : float or numpy.ndarray
        X-position of points to be rotated.
    y : float or numpy.ndarray
        Y-position of points to be rotated.

    Returns
    -------
    xhat : float or numpy.ndarray
        X-positions after rotation.
    yhat : float or numpy.ndarray
        Y-positions after rotation.
    '''
    # Define rotation angle.
    theta = 30 * u.deg

    # Calculate (x, y) in new coordinate system.
    xhat = x * np.cos(theta) + y * np.sin(theta)
    yhat = -x * np.sin(theta) + y * np.cos(theta)

    return xhat, yhat


def rays_to_dispersion(rays, waves, orders):
    '''
    Takes in rays on OGRE focal plane and corresponding waves and
    diffraction orders, then caculates the dispersion on the focal plane
    [Angstrom/mm] for each diffraction order.

    This is used to find the dispersion on the focal plane when
    reconstructing the OGRE spectrum from binned counts on the spectral
    detectors.

    Even though the OGRE focal plane is not the ideal spectrometer focal
    plane, the dispersion is still constant. Therefore, we use the mean
    dispersion value calculated to represent the dispersion [Angstrom/mm]
    for each diffraction order.

    Parameters
    ----------
    rays : [opd, x, y, z, l, m, n, nx, ny, nz]
        Rays on the OGRE focal plane.
    waves : Quantity
        Wavelength values corresponding to each ray.
    order : numpy.ndarray
        Diffraction order values corresponding to each ray.

    Returns
    -------
    n_list : numpy.ndarray
        Array of diffraction orders that dispersion values were
        calculated for.
    disp : list
        List of dispersion Quantities corresponding to dispersion for
        each diffraction order on the OGRE focal plane.
    '''

    # Find unique orders in order array.
    n_list = np.unique(orders)

    # Go through each order and calculate dispersion.
    disp = []
    for n in n_list:
        # Find indices that correspond to this particular order.
        n_ind = np.where(orders == n)
        # Grab particular wavelengths and x-positions.
        waves_n = deepcopy(waves[n_ind])
        x_n = deepcopy(rays[1][n_ind]) * u.mm
        # Remove nan.
        nan_ind = np.isnan(x_n)
        # Caculate mean dispersion.
        # ! This is biased to high wavelengths within spectrum.
        mean_disp = np.mean(waves_n[~nan_ind]/x_n[~nan_ind])
        # Append list.
        disp.append(mean_disp)

    return n_list, disp


def create_spectrum(ray_table, n_list, disp):
    '''
    Creates spectrum from provided ray_table from bin_rays() function
    and calculated dispersion on the focal plane.

    Parameters
    ----------
    ray_table : astropy.Table
        Table of information relating to rays on the OGRE detector.
    n_list : numpy.ndarray
        Array of dispersion values for rays on the OGRE focal plane.
    disp : list
        List of dispersion Quantities for each diffraction order in
        n_list.

    Returns
    -------
    bin_edges : dict
        Dictionary of Quantities corresponding to bin edges in the
        dispersion direction for each order in n_list.
    spect : dict
        Dictionary of values corresponding to each bin in bin_edges for
        each diffraction order. This is the binned spectrum for OGRE.
    '''

    # Move from x-y pixels to dispersion direction.
    x_disp, _ = points_to_dispersion_dir(
        ray_table['x_pix'], ray_table['y_pix'])
    x_disp *= u.mm

    # Add detector center to the dispersion direction positions.
    x_disp += (det_x + 0*u.mm)

    # Get bin edges.
    bin_edges_x = np.arange(-n_pix_x/2, n_pix_x/2 + 1) * pixel_size
    disp_bin_edges, _ = points_to_dispersion_dir(
        bin_edges_x, np.zeros(len(bin_edges_x)))
    disp_bin_edges *= u.mm

    # add detector center to bin edges.
    disp_bin_edges += (det_x + 0*u.mm)

    # Go through each order and create bins.
    bin_edges = {}
    for n, d in zip(n_list, disp):
        # Find bin edges in wavelength space.
        bin_edges['n'+str(n)] = (disp_bin_edges * d).to('Angstrom')

    # Go through each order and create wavelengths.
    spect = {}
    for n, d in zip(n_list, disp):
        # Find which indices correspond to this diffraction order.
        n_ind = np.where(ray_table['order']==n)[0]
        if len(n_ind) < 1:
            # Append zeros if no wavelenths on detector.
            spect['n'+str(n)] = np.zeros(len(disp_bin_edges)-1)
        else:
            # Calculate wavelengths on focal plane for given order.
            wv = (x_disp[n_ind] * d).to('Angstrom')
            # Bin these wavelength values.
            hist, _ = np.histogram(wv, bins=bin_edges['n'+str(n)])
            # Add this histogram to spectrum.
            spect['n'+str(n)] = hist

    return bin_edges, spect


def return_grating_parameters():
    '''
    Returns current grating parameters, including wedge angle
    (relative pitch), center thickness, and yaw.
    '''

    print('Center thicknesses [mm]: ')
    print(np.diff(rg_cen))
    print('')

    print('Average center thickness [mm]: ')
    print(np.mean(np.diff(rg_cen)))
    print('')

    print('Wedge angles (left gratings) [arcmin]: ')
    wedge_ang_left = np.arctan(np.diff(rg_cen/zg_cen_left)).value * u.radian
    print(wedge_ang_left.to('arcmin'))
    print('')

    print('Wedge angles (right gratings) [arcmin]: ')
    wedge_ang_right = np.arctan(np.diff(rg_cen/zg_cen_right)).value * u.radian
    print(wedge_ang_right.to('arcmin'))
    print('')

    print('Average wedge angle [arcmin]: ')
    mean_wedge_ang = (np.mean(wedge_ang_right) + np.mean(wedge_ang_left))/2
    print(mean_wedge_ang.to('arcmin'))
    print('')

    print('Yaw angles (left gratings) [arcmin]: ')
    yaw_left = -yaw + np.arctan((grating_width/2 + cen_width/2) / zg_cen_left).value * u.radian
    print(yaw_left.to('arcmin'))
    print('')

    print('Yaw angles (right gratings) [arcmin]: ')
    yaw_right = -yaw - np.arctan((grating_width/2 + cen_width/2) / zg_cen_right).value * u.radian
    print(yaw_right.to('arcmin'))
    print('')


def return_grating_position_tables():
    '''
    Returns tables giving positions for gratings in the 'left' grating
    stack and in the 'right' grating stack.

    Returns
    -------
    left_table : astropy.Table
        Table specifying the positions of each grating's center in the
        left grating stack.
    right_table : astropy.Table
        Table specifying the positions of each grating's center in the
        right grating stack.
    '''

    # Make and save left grating stack positions.
    x_left = np.ones(len(rg_cen)) * (grating_width/2 + cen_width/2) * u.mm
    left_table = Table(
        [x_left, rg_cen, zg_cen_left], names=['X [mm]', 'Y [mm]', 'Z [mm]'])

    # Make and save right grating stack positions.
    x_right = np.ones(len(rg_cen)) * -1 * (grating_width/2 + cen_width/2) * u.mm
    right_table = Table(
        [x_right, rg_cen, zg_cen_right], names=['X [mm]', 'Y [mm]', 'Z [mm]'])

    return left_table, right_table


def return_relative_grating_stack_orientation():
    '''
    Returns relative orientation of grating stacks within a module. When
    combined with return_grating_position_tables(), this fully defines
    the gratings within a grating module.
    '''

    # Calculate pitch angles of bottom-most grating in each stack.
    # Since gratings are fully determined by wedge angle of grating
    # substrates, this gives relative pitch of entire stack.
    left_pitch = (-rg_cen[0] / zg_cen_left[0]).value + np.radians(1.5)
    right_pitch = (-rg_cen[0] / zg_cen_right[0]).value + np.radians(1.5)
    pitch_diff = left_pitch - right_pitch
    print('Relative pitch (left - right): ' + \
        str(np.degrees(pitch_diff * 60.))+ ' arcmin')

    # Roll of grating. By definition, this is zero.
    print('Relative roll (left - right): 0 arcmin')

    # Relative yaw of stacks. Dependant on where 'center' of grating is.
    # Currently assume center is at grating_width/2
    yaw_left = -yaw + np.arctan(grating_width / zg_cen_left / 2.)
    yaw_right = -yaw - np.arctan(grating_width / zg_cen_right / 2.)
    yaw_diff = np.mean(yaw_left - yaw_right)
    print('Relative yaw (left - right): ' + \
        str(np.degrees(yaw_diff)))


def return_grating_surface_normals():
    '''
    Returns grating surface normals.

    Returns
    -------
    left_table : astropy.Table
        Table specifying the surface normal for each grating in the
        left grating stack.
    right_table : astropy.Table
        Table specifying the surface normal for each grating in the
        right grating stack.
    '''

    # Define initial surface normals.
    l_norm = np.zeros((len(rg_cen), 3))
    l_norm[:,2] = np.ones(len(rg_cen))
    r_norm = deepcopy(l_norm)

    for i in range(len(l_norm)):

        # Rotate to angle of beam.
        R1 = trans.tr.rotation_matrix(-np.pi/2, [1, 0, 0])
        R2 = trans.tr.rotation_matrix(
            -np.arctan((rg_cen[i]/zg_cen_left[i]).to('').value), [1, 0, 0])

        # Put incidence angle onto grating.
        R3 = trans.tr.rotation_matrix(gammy.to('rad').value, [1, 0, 0])

        l_norm[i] = np.dot(R1[:3,:3], l_norm[i])
        l_norm[i] = np.dot(R2[:3,:3], l_norm[i])
        l_norm[i] = np.dot(R3[:3,:3], l_norm[i])

    for i in range(len(r_norm)):

        # Rotate to angle of beam.
        R1 = trans.tr.rotation_matrix(-np.pi/2, [1, 0, 0])
        R2 = trans.tr.rotation_matrix(
            -np.arctan((rg_cen[i]/zg_cen_right[i]).to('').value), [1, 0, 0])

        # Put incidence angle onto grating.
        R3 = trans.tr.rotation_matrix(gammy.to('rad').value, [1, 0, 0])

        r_norm[i] = np.dot(R1[:3, :3], r_norm[i])
        r_norm[i] = np.dot(R2[:3, :3], r_norm[i])
        r_norm[i] = np.dot(R3[:3, :3], r_norm[i])

    return l_norm, r_norm


def return_grating_groove_dir(l_norm, r_norm):
    '''
    Returns direction of central grating groove.

    Returns
    -------
    left_table : astropy.Table
        Table specifying the direction of central grating groove for
        each grating in the left grating stack.
    right_table : astropy.Table
        Table specifying the direction of central grating groove for
        each grating in the right grating stack.
    '''
    # Define initial surface normals.
    l_groove = np.zeros((len(rg_cen), 3))
    l_groove[:, 1] = np.ones(len(rg_cen))
    r_groove = deepcopy(l_groove)

    for i in range(len(l_groove)):

        # Rotate to angle of beam.
        R1 = trans.tr.rotation_matrix(-np.pi/2, [1, 0, 0])
        R2 = trans.tr.rotation_matrix(
            -np.arctan((rg_cen[i]/zg_cen_left[i]).to('').value), [1, 0, 0])

        # Put incidence angle onto grating.
        R3 = trans.tr.rotation_matrix(gammy.to('rad').value, [1, 0, 0])

        # Add yaw.
        yaw_add = np.arctan(
            ((grating_width/2 + cen_width/2) / zg_cen_left[i]).to('').value)
        R4 = trans.tr.rotation_matrix(-yaw.to('rad').value + yaw_add, l_norm[i])

        l_groove[i] = np.dot(R1[:3, :3], l_groove[i])
        l_groove[i] = np.dot(R2[:3, :3], l_groove[i])
        l_groove[i] = np.dot(R3[:3, :3], l_groove[i])
        l_groove[i] = np.dot(R4[:3, :3], l_groove[i])

    for i in range(len(r_groove)):

        # Rotate to angle of beam.
        R1 = trans.tr.rotation_matrix(-np.pi/2, [1, 0, 0])
        R2 = trans.tr.rotation_matrix(
            -np.arctan((rg_cen[i]/zg_cen_right[i]).to('').value), [1, 0, 0])

        # Put incidence angle onto grating.
        R3 = trans.tr.rotation_matrix(gammy.to('rad').value, [1, 0, 0])

        yaw_add = np.arctan(
            ((grating_width + cen_width) / 2 / zg_cen_right[i]).to('').value)
        R4 = trans.tr.rotation_matrix(-yaw.to('rad').value - yaw_add, r_norm[i])

        r_groove[i] = np.dot(R1[:3, :3], r_groove[i])
        r_groove[i] = np.dot(R2[:3, :3], r_groove[i])
        r_groove[i] = np.dot(R3[:3, :3], r_groove[i])
        r_groove[i] = np.dot(R4[:3, :3], r_groove[i])

    return l_groove, r_groove


def f(x, A, B):
    '''Line function for bin_rays_simplified()'''
    return A*x + B


def bin_rays_simplified(rays, waves, orders, oa_misalign,
                        g_misalign, s_misalign, m_misalign, det_z):

    # Calculate min and max detector positions
    disp = 1. / groove_period / (l_hub / np.cos(gamma))
    det_min = (det_x - 12.7*u.mm) * disp
    det_max = (det_x + 12.7*u.mm) * disp

    # Calculate varying dispersion for rays.
    test_wvs = np.linspace(det_min, det_max, 20)
    disp = []
    r = []

    for test_wv in test_wvs:
        # Simulate OGRE channel
        o = np.ones(10000)
        w = np.copy(o) * test_wv
        temp_rays, _, _ = os.ogre_simulation_one_channel(w, o, oa_misalign,
                                                    g_misalign, s_misalign,
                                                    m_misalign, det_z)
        disp.append(np.mean(temp_rays[1]))
        r.append(np.mean(temp_rays[1])/2.355/np.std(temp_rays[1]))

    # Move to dispersion [mm/nm]
    disp = np.array(disp) * u.mm
    # dispersion = test_wvs.to(u.Angstrom) / (np.array(disp) * u.mm)

    # Create dispersion fit.
    det_bins = np.arange((det_x - 12.7*u.mm).value,
                         (det_x + 12.7*u.mm).value + 16e-3,
                         16e-3)
    popt, pcov = curve_fit(f, disp.to(u.mm).value,
                           test_wvs.to(u.angstrom).value)

    # Go through and create spectra.
    spect = {}
    wv_bins = {}
    wave_bins = f(det_bins, popt[0], popt[1])
    for n in [1, 2, 3, 4]:
        # Find indices for this specific order.
        n_ind = np.where(orders == n)
        # Create bins.
        spect['n'+str(n)] = np.histogram(rays[1][n_ind], bins=det_bins)[0]
        wv_bins['n'+str(n)] = wave_bins / n

    return wv_bins, spect
