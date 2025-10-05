# -*- coding: utf-8 -*-
# Original BigFISH authorship: Arthur Imbert <arthur.imbert.pro@gmail.com> (base bigfish 2D implementation)
# 3D extension developed by Saintgene Xu <saintgene@gmail.com>
# License: BSD 3-Clause
#
# NOTE
# -----
# This module is a 3D extension of the BigFISH 2D feature computation, keeping
# identical feature names/definitions while operating on volumetric data (z, y, x).
# Distances are computed in physical units (nanometers) by using the `sampling`
# parameter of `scipy.ndimage.distance_transform_edt`, which naturally supports
# anisotropic voxels.
#
# Inputs (conventions):
# - rna_coord: shape (N, 4). The first three columns are (z, y, x) indices;
#   the last column is the cluster id (or -1 if not available).
# - cell_mask / nuc_mask: 3D boolean arrays (Z, Y, X).
# - smfish: 3D image (Z, Y, X).
# - All "area"-like quantities from the 2D version become "volume" in 3D.

import numpy as np
from scipy import ndimage as ndi

# import skimage
# from sklearn.utils.fixes import parse_version
# if parse_version(skimage.__version__) < parse_version("0.19.0"):
#     from skimage.morphology.selem import disk, ball
# else:
#     from skimage.morphology.footprints import disk, ball

import bigfish.stack as stack
from .input_preparation3d import prepare_extracted_data3d


def compute_features3d(
        cell_mask,
        nuc_mask,
        rna_coord,
        smfish=None,
        voxel_size_zyx=None,  # (vz, vy, vx) in nanometers
        foci_coord=None,
        centrosome_coord=None,
        compute_distance=False,
        compute_intranuclear=False,
        compute_protrusion=False,
        compute_dispersion=False,
        compute_topography=False,
        compute_foci=False,
        compute_area=False,
        compute_centrosome=False,
        return_names=False):
    """Compute spatial features on 3D volumes (same semantics as the 2D version).

    Parameters
    ----------
    cell_mask : (Z, Y, X) bool
    nuc_mask : (Z, Y, X) bool
    rna_coord : (N, 4) int
        Each row is [z, y, x, cluster_id]; if clustering is unavailable, set cluster_id = -1.
    smfish : (Z, Y, X) uint, optional
    voxel_size_zyx : tuple(float, float, float)
        Physical voxel size in nanometers, passed to EDT to obtain physical distances.
    foci_coord : (F, 5) int, optional
        Each row is [z, y, x, n_spots_in_foci, foci_index].
    centrosome_coord : (C, 3) int, optional
        Each row is [z, y, x]; multiple centrosomes are allowed.
    compute_* : bool
        Flags to control which feature blocks are computed.
    return_names : bool
        Whether to also return the list of feature names.

    Returns
    -------
    features : np.ndarray (float32)
    (features_names : list[str]) if return_names is True
    """
    # Parameter checks
    stack.check_parameter(
        voxel_size_zyx=(tuple, list, type(None)),
        compute_distance=bool,
        compute_intranuclear=bool,
        compute_protrusion=bool,
        compute_dispersion=bool,
        compute_topography=bool,
        compute_foci=bool,
        compute_area=bool,
        compute_centrosome=bool,
        return_names=bool)

    stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool, np.uint8, np.uint16, np.int32, np.int64])
    stack.check_array(nuc_mask, ndim=3, dtype=[np.bool_, bool, np.uint8, np.uint16, np.int32, np.int64])
    cell_mask = cell_mask.astype(bool)
    nuc_mask = nuc_mask.astype(bool)

    if smfish is not None:
        stack.check_array(smfish, ndim=3, dtype=[np.uint8, np.uint16])

    stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
    if rna_coord.shape[1] < 3:
        raise ValueError("`rna_coord` must contain at least 3 columns (z, y, x).")

    if foci_coord is not None:
        stack.check_array(foci_coord, ndim=2, dtype=[np.int32, np.int64])

    if centrosome_coord is not None:
        stack.check_array(centrosome_coord, ndim=2, dtype=[np.int32, np.int64])

    if voxel_size_zyx is None:
        raise ValueError("'voxel_size_zyx' must be provided as (vz, vy, vx) in nanometers.")
    if len(voxel_size_zyx) != 3:
        raise ValueError("'voxel_size_zyx' should be a 3-tuple (vz, vy, vx).")
    vz, vy, vx = map(float, voxel_size_zyx)

    # Prepare all derived quantities in a single call (3D distance maps, centroids, etc.)
    (cell_mask,
     distance_cell, distance_cell_normalized,
     centroid_cell, distance_centroid_cell,
     nuc_mask, cell_mask_out_nuc,
     distance_nuc, distance_nuc_normalized,
     centroid_nuc, distance_centroid_nuc,
     rna_coord_out_nuc,
     centroid_rna, distance_centroid_rna,
     centroid_rna_out_nuc, distance_centroid_rna_out_nuc,
     distance_centrosome) = prepare_extracted_data3d(
        cell_mask, nuc_mask, rna_coord, centrosome_coord, voxel_size_zyx)

    features = ()
    names = {
        'distance': False,
        'intranuclear': False,
        'protrusion': False,
        'dispersion': False,
        'topography': False,
        'foci': False,
        'area': False,
        'centrosome': False,
    }

    if compute_distance:
        features += features_distance3d(
            rna_coord, distance_cell, distance_nuc, cell_mask, check_input=False)
        names['distance'] = True

    if compute_intranuclear:
        features += features_in_out_nucleus3d(
            rna_coord, rna_coord_out_nuc, check_input=False)
        names['intranuclear'] = True

    if compute_protrusion:
        features += features_protrusion3d(
            rna_coord, cell_mask, nuc_mask, voxel_size_zyx, check_input=False)
        names['protrusion'] = True

    if compute_dispersion:
        # if smfish is None:
        #     raise ValueError("A 3D smFISH image is required for dispersion/polarization indices.")
        features += features_dispersion3d(
            smfish, rna_coord, centroid_rna, cell_mask, centroid_cell, centroid_nuc, check_input=False)
        names['dispersion'] = True

    if compute_topography:
        features += features_topography3d(
            rna_coord, cell_mask, nuc_mask, cell_mask_out_nuc, voxel_size_zyx, check_input=False)
        names['topography'] = True

    if compute_foci:
        if foci_coord is None:
            raise ValueError("`foci_coord` is required to compute foci-related features.")
        features += features_foci3d(rna_coord, foci_coord, check_input=False)
        names['foci'] = True

    if compute_area:
        features += features_area3d(cell_mask, nuc_mask, cell_mask_out_nuc, voxel_size_zyx, check_input=False)
        names['area'] = True

    if compute_centrosome:
        if (centrosome_coord is None) or (smfish is None):
            raise ValueError("'centrosome_coord' and 'smfish' are required to compute centrosome-related features.")
        features += features_centrosome3d(
            smfish, rna_coord, distance_centrosome, cell_mask, voxel_size_zyx, check_input=False)
        names['centrosome'] = True

    features = np.array(features, dtype=np.float32)
    features = np.round(features, 2)

    if return_names:
        names_list = get_features_name3d(**{f"names_features_{k}": v for k, v in [
            ("distance", names['distance']),
            ("intranuclear", names['intranuclear']),
            ("protrusion", names['protrusion']),
            ("dispersion", names['dispersion']),
            ("topography", names['topography']),
            ("foci", names['foci']),
            ("area", names['area']),
            ("centrosome", names['centrosome']),
        ]})
        return features, names_list

    return features


def get_features_name3d(
        names_features_distance=False,
        names_features_intranuclear=False,
        names_features_protrusion=False,
        names_features_dispersion=False,
        names_features_topography=False,
        names_features_foci=False,
        names_features_area=False,
        names_features_centrosome=False):
    """Return the 3D feature-name list (aligned with the 2D naming)."""
    stack.check_parameter(
        names_features_distance=bool,
        names_features_intranuclear=bool,
        names_features_protrusion=bool,
        names_features_dispersion=bool,
        names_features_topography=bool,
        names_features_foci=bool,
        names_features_area=bool,
        names_features_centrosome=bool)

    features_name = []

    if names_features_distance:
        features_name += [
            "index_mean_distance_cell",
            "index_median_distance_cell",
            "index_mean_distance_nuc",
            "index_median_distance_nuc",
        ]

    if names_features_intranuclear:
        features_name += [
            "proportion_rna_in_nuc",
            "nb_rna_out_nuc",
            "nb_rna_in_nuc",
        ]

    if names_features_protrusion:
        features_name += [
            "index_rna_protrusion",
            "proportion_rna_protrusion",
            "protrusion_volume",
        ]

    if names_features_dispersion:
        features_name += [
            "index_polarization",
            "index_dispersion",
            "index_peripheral_distribution",
        ]

    if names_features_topography:
        features_name += [
            "index_rna_nuc_edge",
            "proportion_rna_nuc_edge",
        ]
        a = 500
        for b in range(1000, 3001, 500):
            features_name += [
                f"index_rna_nuc_radius_{a}_{b}",
                f"proportion_rna_nuc_radius_{a}_{b}",
            ]
            a = b

        a = 0
        for b in range(500, 3001, 500):
            features_name += [
                f"index_rna_cell_radius_{a}_{b}",
                f"proportion_rna_cell_radius_{a}_{b}",
            ]
            a = b

    if names_features_foci:
        features_name += ["proportion_rna_in_foci"]

    if names_features_area:
        features_name += [
            "proportion_nuc_volume",
            "cell_volume",
            "nuc_volume",
            "cell_volume_out_nuc",
        ]

    if names_features_centrosome:
        features_name += [
            "index_mean_distance_centrosome",
            "index_median_distance_centrosome",
            "index_rna_centrosome",
            "proportion_rna_centrosome",
            "index_centrosome_dispersion",
        ]

    return features_name


# -------------------- Feature implementations (3D) --------------------

def _rna_index(rna_coord):
    """Return the first three columns (z, y, x) for 3D indexing."""
    return (rna_coord[:, 0], rna_coord[:, 1], rna_coord[:, 2])


def features_distance3d(rna_coord, distance_cell, distance_nuc, cell_mask, check_input=True):
    """Distance-based indices: sample physical distance maps at RNA positions and
    normalize by the expectation within the cell mask."""
    if check_input:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(distance_cell, ndim=3, dtype=[np.float16, np.float32, np.float64])
        stack.check_array(distance_nuc, ndim=3, dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool])

    if len(rna_coord) == 0:
        return (1., 1., 1., 1.)

    z, y, x = _rna_index(rna_coord)
    rna_distance_cell = distance_cell[z, y, x]
    expected_distance = AboveZero(float(np.mean(distance_cell[cell_mask])))
    index_mean_dist_cell = float(np.mean(rna_distance_cell)) / expected_distance
    expected_distance = AboveZero(float(np.median(distance_cell[cell_mask])))
    index_median_dist_cell = float(np.median(rna_distance_cell)) / expected_distance

    rna_distance_nuc = distance_nuc[z, y, x]
    expected_distance = AboveZero(float(np.mean(distance_nuc[cell_mask])))
    index_mean_dist_nuc = float(np.mean(rna_distance_nuc)) / expected_distance
    expected_distance = AboveZero(float(np.median(distance_nuc[cell_mask])))
    index_median_dist_nuc = float(np.median(rna_distance_nuc)) / expected_distance

    return (index_mean_dist_cell, index_median_dist_cell,
            index_mean_dist_nuc, index_median_dist_nuc)


def features_in_out_nucleus3d(rna_coord, rna_coord_out_nuc, check_input=True):
    """In/out nucleus: return proportion in nucleus and the in/out counts."""
    if check_input:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(rna_coord_out_nuc, ndim=2, dtype=[np.int32, np.int64])

    nb_rna = float(len(rna_coord))
    if nb_rna == 0:
        return (0., 0., 0.)

    nb_rna_out_nuc = float(len(rna_coord_out_nuc))
    nb_rna_in_nuc = nb_rna - nb_rna_out_nuc
    proportion_rna_in_nuc = nb_rna_in_nuc / nb_rna

    return (proportion_rna_in_nuc, nb_rna_out_nuc, nb_rna_in_nuc)


def features_protrusion3d(rna_coord, cell_mask, nuc_mask, voxel_size_zyx, check_input=True):
    """Protrusion features (3D): approximate removal of thin protrusions using a
    3D EDT-based opening; treat the nucleus as solid to avoid misclassifying it as protrusion."""
    if check_input:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_array(nuc_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_parameter(voxel_size_zyx=(tuple, list))

    nb_rna = len(rna_coord)
    cell_volume = int(cell_mask.sum())

    vox_vol = np.prod(voxel_size_zyx)/1e9
    # Use physical radius r_nm in an EDT-based opening to remove thin protrusions.
    mask_opened = opening_by_edt(cell_mask, r_nm=3000.0, voxel_size_zyx=voxel_size_zyx)
    # Make the nuclear region solid (True) to avoid labeling it as protrusion.
    mask_opened[nuc_mask] = True
    protrusion_volume = (cell_volume - int(mask_opened.sum()))*vox_vol # um^3
    
    if nb_rna == 0:
        return (1., 0., float(protrusion_volume))

    if protrusion_volume > 0:
        z, y, x = _rna_index(rna_coord)
        kept = mask_opened[z, y, x]  # True means still in the "non-protrusion" region
        nb_rna_protrusion = int(nb_rna - np.count_nonzero(kept))
        expected_rna_protrusion = nb_rna * (protrusion_volume / float(cell_volume))
        index_rna_protrusion = nb_rna_protrusion / expected_rna_protrusion if expected_rna_protrusion > 0 else 1.0
        proportion_rna_protrusion = nb_rna_protrusion / float(nb_rna)
        return (float(index_rna_protrusion), float(proportion_rna_protrusion), float(protrusion_volume))
    else:
        return (1., 0., 0.)


def features_dispersion3d(smfish, rna_coord, centroid_rna, cell_mask, centroid_cell, centroid_nuc, check_input=True):
    """3D version of the RDI triplet: polarization, dispersion, and peripheral distribution indices."""
    if check_input:
        stack.check_array(smfish, ndim=3, dtype=[np.uint8, np.uint16])
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(centroid_rna, ndim=1, dtype=[np.float32, np.float64, np.int32, np.int64])
        stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_array(centroid_cell, ndim=1, dtype=[np.float32, np.float64, np.int32, np.int64])
        stack.check_array(centroid_nuc, ndim=1, dtype=[np.float32, np.float64, np.int32, np.int64])

    if len(rna_coord) == 0:
        return (0., 1., 1.)

    # All coordinates are (z, y, x)
    centroid_rna_ = centroid_rna.astype(np.float64)
    centroid_cell_ = centroid_cell.astype(np.float64)
    centroid_nuc_ = centroid_nuc.astype(np.float64)

    # Cell voxel coordinates and intensities
    cell_idx = np.nonzero(cell_mask)
    cell_coord = np.column_stack(cell_idx)  # (M, 3)
    if smfish is not None:
        cell_value = smfish[cell_idx]
    else:
        cell_value = np.ones(cell_coord.shape[0])
            
    total_intensity_cell = float(cell_value.sum()) if cell_value.size > 0 else 1.0

    # RNA voxel positions and intensities (sample smFISH intensities at RNA voxels)
    z, y, x = _rna_index(rna_coord)
    rna_mask = np.zeros_like(cell_mask, dtype=bool)
    rna_mask[z, y, x] = True
    rna_idx = np.nonzero(rna_mask)
    rna_coord_vox = np.column_stack(rna_idx)
    if smfish is not None:
        rna_value = smfish[rna_idx]
    else:
        rna_value = np.ones(rna_coord.shape[0])
        
    total_intensity_rna = float(rna_value.sum()) if rna_value.size > 0 else 1.0

    # 1) PI: distance between RNA COM and cell centroid divided by the radius of gyration of the cell point cloud
    centroid_distance = float(np.linalg.norm(centroid_rna_ - centroid_cell_))
    gyration_radius = _rmsd3d(cell_coord, centroid_cell_)
    index_polarization = centroid_distance / gyration_radius if gyration_radius > 0 else 0.0

    # 2) DI: intensity-weighted mean squared distance around the RNA COM, normalized by the same metric for all cell voxels
    r = np.linalg.norm(rna_coord_vox - centroid_rna_, axis=1) ** 2
    a = float(np.sum((r * rna_value) / total_intensity_rna))
    r = np.linalg.norm(cell_coord - centroid_rna_, axis=1) ** 2
    b = float(np.sum((r * cell_value) / total_intensity_cell))
    index_dispersion = a / b if b > 0 else 1.0

    # 3) PDI: same as DI but referenced to the nuclear centroid (nuclear-distal bias)
    r = np.linalg.norm(rna_coord_vox - centroid_nuc_, axis=1) ** 2
    a = float(np.sum((r * rna_value) / total_intensity_rna))
    r = np.linalg.norm(cell_coord - centroid_nuc_, axis=1) ** 2
    b = float(np.sum((r * cell_value) / total_intensity_cell))
    index_peripheral_distribution = a / b if b > 0 else 1.0

    return (float(index_polarization), float(index_dispersion), float(index_peripheral_distribution))


def _rmsd3d(coord, reference_coord):
    n = len(coord)
    if n == 0:
        return 0.0
    diff = coord - reference_coord
    return float(np.sqrt((diff ** 2).sum() / n))


def features_topography3d(rna_coord, cell_mask, nuc_mask, cell_mask_out_nuc, voxel_size_zyx, check_input=True):
    """Topographic features (3D): build nuclear- and cell-edge shells at 500 nm steps,
    and compute enrichment and proportion within each shell."""
    if check_input:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_array(nuc_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_array(cell_mask_out_nuc, ndim=3, dtype=[np.bool_, bool])
        stack.check_parameter(voxel_size_zyx=(tuple, list))

    nb_rna = len(rna_coord)
    if nb_rna == 0:
        features = (0., 0.)
        features += (0., 0.) * 5
        features += (0., 0.) * 6
        return features

    # Build physical distance maps (nm)
    vz, vy, vx = map(float, voxel_size_zyx)
    sampling = (vz, vy, vx)

    # Distance to the nuclear "boundary" (both inside and outside sides)
    dist_nuc_out = ndi.distance_transform_edt(~nuc_mask, sampling=sampling)
    dist_nuc_in = ndi.distance_transform_edt(~cell_mask_out_nuc, sampling=sampling)
    dist_nuc = dist_nuc_out + dist_nuc_in
    dist_nuc[~cell_mask] = 0.0

    # Distance to the cell membrane (cell_mask voxels have distance 0 to membrane)
    dist_cell = ndi.distance_transform_edt(cell_mask, sampling=sampling)

    cell_volume = int(cell_mask.sum())
    z, y, x = _rna_index(rna_coord)

    # 1) Nuclear edge < 500 nm shell
    thresh = 500.0  # nm
    shell = dist_nuc < thresh
    shell[~cell_mask] = False
    shell_volume = max(int(shell.sum()), 1)
    expected = nb_rna * (shell_volume / float(cell_volume))
    observed = int(np.count_nonzero(shell[z, y, x]))
    index_edge = observed / expected if expected > 0 else 1.0
    prop_edge = observed / float(nb_rna)
    features = (float(index_edge), float(prop_edge))

    # 2) Five nuclear shells from 500–3000 nm
    cum = shell.copy()
    for i in range(2, 7):
        shell_i = dist_nuc < (i * thresh)
        shell_i[~cell_mask] = False
        shell_i[nuc_mask] = False
        shell_i[cum] = False
        cum |= shell_i
        vol = max(int(shell_i.sum()), 1)
        expected = nb_rna * (vol / float(cell_volume))
        observed = int(np.count_nonzero(shell_i[z, y, x]))
        features += (float(observed / expected if expected > 0 else 1.0),
                     float(observed / float(nb_rna)))

    # 3) Six cell-edge shells from 0–3000 nm
    cum = np.zeros_like(cell_mask, dtype=bool)
    for i in range(1, 7):
        shell_i = dist_cell < (i * thresh)
        shell_i[~cell_mask] = False
        shell_i[nuc_mask] = False
        shell_i[cum] = False
        cum |= shell_i
        vol = max(int(shell_i.sum()), 1)
        expected = nb_rna * (vol / float(cell_volume))
        observed = int(np.count_nonzero(shell_i[z, y, x]))
        features += (float(observed / expected if expected > 0 else 1.0),
                     float(observed / float(nb_rna)))

    return features


def features_foci3d(rna_coord, foci_coord, check_input=True):
    """Foci features (3D): use the fourth column (n_spots_in_foci) to compute the proportion in foci."""
    if check_input:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(foci_coord, ndim=2, dtype=[np.int32, np.int64])

    if (len(rna_coord) == 0) or (len(foci_coord) == 0):
        return (0.,)

    nb_rna = float(len(rna_coord))
    # Convention: the 4th column (index=3) in foci_coord stores the number of spots in the foci
    nb_rna_in_foci = int(foci_coord[:, 3].sum()) if foci_coord.shape[1] >= 4 else 0
    return (float(nb_rna_in_foci / nb_rna),)


def features_area3d(cell_mask, nuc_mask, cell_mask_out_nuc, voxel_size_zyx, check_input=True):
    """Volume-related features: nuclear volume fraction and absolute volumes (µm^3)."""
    if check_input:
        stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_array(nuc_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_array(cell_mask_out_nuc, ndim=3, dtype=[np.bool_, bool])
        stack.check_parameter(voxel_size_zyx=(tuple, list))

    vox_vol = np.prod(voxel_size_zyx)/1e9
    cell_vol = float(cell_mask.sum())
    nuc_vol = float(nuc_mask.sum())
    rel = nuc_vol / cell_vol if cell_vol > 0 else 0.0
    cell_vol_out_nuc = float(cell_mask_out_nuc.sum())
    return (float(rel), float(cell_vol*vox_vol), float(nuc_vol*vox_vol), float(cell_vol_out_nuc*vox_vol))


def features_centrosome3d(smfish, rna_coord, distance_centrosome, cell_mask, voxel_size_zyx, check_input=True):
    """Centrosome-related features (3D): normalized mean/median distance, 2000‑nm
    neighborhood enrichment and proportion, and a dispersion index."""
    if check_input:
        stack.check_array(smfish, ndim=3, dtype=[np.uint8, np.uint16])
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
        stack.check_array(distance_centrosome, ndim=3, dtype=[np.float16, np.float32, np.float64])
        stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool])
        stack.check_parameter(voxel_size_zyx=(tuple, list))

    if len(rna_coord) == 0:
        return (1., 1., 1., 0., 1.)

    nb_rna = len(rna_coord)
    cell_vol = int(cell_mask.sum())
    z, y, x = _rna_index(rna_coord)

    # Normalize distances using cell interior expectation
    rna_d = distance_centrosome[z, y, x]
    expected = float(np.mean(distance_centrosome[cell_mask]))
    idx_mean = float(np.mean(rna_d)) / expected if expected > 0 else 1.0
    expected = float(np.median(distance_centrosome[cell_mask]))
    idx_med = float(np.median(rna_d)) / expected if expected > 0 else 1.0

    features = (idx_mean, idx_med)

    # 2000 nm neighborhood (distance map is already in nm)
    radius_nm = 2000.0
    mask_cent = distance_centrosome < radius_nm
    mask_cent[~cell_mask] = False
    vol = max(int(mask_cent.sum()), 1)
    expected_nb = nb_rna * (vol / float(cell_vol))
    observed_nb = int(np.count_nonzero(mask_cent[z, y, x]))
    idx_enrich = observed_nb / expected_nb if expected_nb > 0 else 1.0
    prop = observed_nb / float(nb_rna)
    features += (float(idx_enrich), float(prop))

    # Dispersion index with respect to the centrosome (intensity-weighted r^2)
    cell_idx = np.nonzero(cell_mask)
    cell_coord = np.column_stack(cell_idx)
    cell_val = smfish[cell_idx]
    I_cell = float(cell_val.sum()) if cell_val.size > 0 else 1.0

    rna_val = smfish[z, y, x]
    I_rna = float(rna_val.sum()) if rna_val.size > 0 else 1.0

    r2_rna = (distance_centrosome[z, y, x] ** 2)
    a = float(np.sum((r2_rna * rna_val) / I_rna))

    r2_cell = (distance_centrosome[cell_idx] ** 2)
    b = float(np.sum((r2_cell * cell_val) / I_cell))
    idx_disp = a / b if b > 0 else 1.0

    features += (float(idx_disp),)
    return features


def opening_by_edt(mask, r_nm, voxel_size_zyx):
    """Binary opening implemented via EDT in physical space:
    1) erode: keep voxels with distance-to-background >= r_nm;
    2) dilate: keep voxels with distance-to-eroded <= r_nm."""
    edt = ndi.distance_transform_edt(mask, sampling=voxel_size_zyx).astype(np.float32)
    eroded = edt >= r_nm
    edt2 = ndi.distance_transform_edt(~eroded, sampling=voxel_size_zyx).astype(np.float32)
    opened = edt2 <= r_nm
    return opened

def AboveZero(val_In):
    """Avoid division by zero in index normalization by clamping zeros to 1.0."""
    if val_In == 0:
        val_Out = 1.0
    else:
        val_Out = val_In
    return val_Out
