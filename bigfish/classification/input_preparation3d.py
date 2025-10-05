# -*- coding: utf-8 -*-
# Original BigFISH authorship: Arthur Imbert <arthur.imbert.pro@gmail.com> (base bigfish 2D implementation)
# 3D extension developed by Saintgene Xu <saintgene@gmail.com>
# License: BSD 3-Clause
#
# 3D input preparation:
# This module generalizes BigFISH's `prepare_extracted_data` to volumetric data
# and builds distance maps in physical units (nanometers) via the `sampling`
# parameter of `scipy.ndimage.distance_transform_edt`.

import numpy as np
from scipy import ndimage as ndi

import bigfish.stack as stack

from skimage.measure import regionprops


def prepare_extracted_data3d(
        cell_mask,
        nuc_mask=None,
        rna_coord=None,
        centrosome_coord=None,
        voxel_size_zyx=None):
    """Prepare all derived quantities required by 3D feature computation.

    Parameters
    ----------
    cell_mask : (Z, Y, X) bool
    nuc_mask : (Z, Y, X) bool or None
    rna_coord : (N, >=3) int or None
        First three columns are (z, y, x).
    centrosome_coord : (C, 3) int or None
    voxel_size_zyx : tuple(float, float, float)
        Physical voxel size (nanometers).

    Returns
    -------
    A tuple of 17 elements, aligned with the 2D version (where "area" becomes "volume" in 3D):
    (cell_mask,
     distance_cell, distance_cell_normalized,
     centroid_cell, distance_centroid_cell,
     nuc_mask, cell_mask_out_nuc,
     distance_nuc, distance_nuc_normalized,
     centroid_nuc, distance_centroid_nuc,
     rna_coord_out_nuc,
     centroid_rna, distance_centroid_rna,
     centroid_rna_out_nuc, distance_centroid_rna_out_nuc,
     distance_centrosome)
    """
    stack.check_array(cell_mask, ndim=3, dtype=[np.bool_, bool, np.uint8, np.uint16, np.int32, np.int64])
    cell_mask = cell_mask.astype(bool)
    if nuc_mask is not None:
        stack.check_array(nuc_mask, ndim=3, dtype=[np.bool_, bool, np.uint8, np.uint16, np.int32, np.int64])
        nuc_mask = nuc_mask.astype(bool)
    if rna_coord is not None:
        stack.check_array(rna_coord, ndim=2, dtype=[np.int32, np.int64])
    if centrosome_coord is not None:
        stack.check_array(centrosome_coord, ndim=2, dtype=[np.int32, np.int64])

    if voxel_size_zyx is None:
        raise ValueError("'voxel_size_zyx' must be provided as (vz, vy, vx) in nanometers.")
    vz, vy, vx = map(float, voxel_size_zyx)
    sampling = (vz, vy, vx)

    # Distance to membrane (nm)
    distance_cell = ndi.distance_transform_edt(cell_mask, sampling=sampling).astype(np.float32)
    distance_cell_normalized = distance_cell / (distance_cell.max() if distance_cell.max() > 0 else 1.0)

    # Cell centroid (geometric center, integer voxel indices) and its distance map (nm)
    centroid_cell = _get_centroid_surface3d(cell_mask).astype(np.int64)
    distance_centroid_cell = _get_centroid_distance_map3d(centroid_cell, cell_mask, sampling)

    # Nucleus-related quantities
    if nuc_mask is not None:
        cell_mask_out_nuc = cell_mask.copy()
        cell_mask_out_nuc[nuc_mask] = False

        dist_nuc_out = ndi.distance_transform_edt(~nuc_mask, sampling=sampling)
        distance_nuc = (cell_mask * dist_nuc_out).astype(np.float32)
        distance_nuc_normalized = distance_nuc / (distance_nuc.max() if distance_nuc.max() > 0 else 1.0)

        centroid_nuc = _get_centroid_surface3d(nuc_mask).astype(np.int64)
        distance_centroid_nuc = _get_centroid_distance_map3d(centroid_nuc, cell_mask, sampling)
    else:
        cell_mask_out_nuc = None
        distance_nuc = None
        distance_nuc_normalized = None
        centroid_nuc = None
        distance_centroid_nuc = None

    # RNA-related quantities
    if rna_coord is not None:
        if len(rna_coord) == 0:
            centroid_rna = np.array([0, 0, 0], dtype=np.int64)
        else:
            centroid_rna = _get_centroid_rna3d(rna_coord)

        distance_centroid_rna = _get_centroid_distance_map3d(centroid_rna, cell_mask, sampling)

        if nuc_mask is not None:
            z, y, x = rna_coord[:, 0], rna_coord[:, 1], rna_coord[:, 2]
            mask_in_nuc = nuc_mask[z, y, x]
            rna_coord_out_nuc = rna_coord[~mask_in_nuc]

            if len(rna_coord_out_nuc) == 0:
                centroid_rna_out_nuc = np.array([0, 0, 0], dtype=np.int64)
            else:
                centroid_rna_out_nuc = _get_centroid_rna3d(rna_coord_out_nuc)

            distance_centroid_rna_out_nuc = _get_centroid_distance_map3d(centroid_rna_out_nuc, cell_mask, sampling)
        else:
            rna_coord_out_nuc = None
            centroid_rna_out_nuc = None
            distance_centroid_rna_out_nuc = None
    else:
        centroid_rna = None
        distance_centroid_rna = None
        rna_coord_out_nuc = None
        centroid_rna_out_nuc = None
        distance_centroid_rna_out_nuc = None

    # Distance to the nearest centrosome (nm)
    if centrosome_coord is not None:
        if len(centrosome_coord) == 0:
            distance_centrosome = distance_cell.copy()
        else:
            distance_centrosome = _get_centrosome_distance_map3d(centrosome_coord, cell_mask, sampling)
    else:
        distance_centrosome = None

    return (cell_mask,
            distance_cell, distance_cell_normalized,
            centroid_cell, distance_centroid_cell,
            nuc_mask, cell_mask_out_nuc,
            distance_nuc, distance_nuc_normalized,
            centroid_nuc, distance_centroid_nuc,
            rna_coord_out_nuc,
            centroid_rna, distance_centroid_rna,
            centroid_rna_out_nuc, distance_centroid_rna_out_nuc,
            distance_centrosome)


def _get_centroid_surface3d(mask):
    """Geometric centroid (z, y, x) of a 3D binary object."""
    lab = mask.astype(np.uint8)
    if lab.sum() == 0:
        return np.array([0, 0, 0], dtype=np.int64)
    region = regionprops(lab)[0]
    cz, cy, cx = region.centroid  # floats
    return np.array([int(round(cz)), int(round(cy)), int(round(cx))], dtype=np.int64)


def _get_centroid_rna3d(rna_coord):
    """Geometric centroid of RNA voxel coordinates (mean over the first 3 columns z, y, x)."""
    return np.mean(rna_coord[:, :3], axis=0, dtype=rna_coord.dtype)


def _get_centroid_distance_map3d(centroid_zyx, cell_mask, sampling):
    """Construct a 3D physical distance map (nm) to the given centroid within the cell mask."""
    z, y, x = map(int, np.round(centroid_zyx))
    mask_centroid = np.zeros_like(cell_mask, dtype=bool)
    # Defensive clipping to image bounds
    z = np.clip(z, 0, cell_mask.shape[0] - 1)
    y = np.clip(y, 0, cell_mask.shape[1] - 1)
    x = np.clip(x, 0, cell_mask.shape[2] - 1)
    mask_centroid[z, y, x] = True
    dist = ndi.distance_transform_edt(~mask_centroid, sampling=sampling).astype(np.float32)
    dist[~cell_mask] = 0.0
    return dist


def _get_centrosome_distance_map3d(centrosome_coord, cell_mask, sampling):
    """For multiple centrosomes, return the 3D nearest-distance map (nm)."""
    mask_cent = np.zeros_like(cell_mask, dtype=bool)
    z = np.clip(centrosome_coord[:, 0], 0, cell_mask.shape[0] - 1)
    y = np.clip(centrosome_coord[:, 1], 0, cell_mask.shape[1] - 1)
    x = np.clip(centrosome_coord[:, 2], 0, cell_mask.shape[2] - 1)
    mask_cent[z, y, x] = True
    dist = ndi.distance_transform_edt(~mask_cent, sampling=sampling).astype(np.float32)
    dist[~cell_mask] = 0.0
    return dist
