# -*- coding: utf-8 -*-
"""
Parallel + round-level caching (joblib + cached)
Process_RNALocs_Features_joblib_cached.py

Goal: on top of cell-level parallelization (one CID per worker), cache the
round-level derived quantities (distance maps, centroids, protrusion opening)
once per round and reuse them across the six genes in the same round, which
significantly reduces redundant 3D EDT computations. The feature logic and
ordering match `features3d.compute_features3d` exactly.

Developed by Saintgene Xu <saintgene@gmail.com>
License: BSD 3-Clause

Dependencies
------------
- `features3d.py`, `input_preparation3d.py` (your 3D-extended versions in this repo)
- bigfish.stack, scipy.ndimage

Usage
-----
    python Process_RNALocs_Features_joblib_cached.py -j -1 --rc 5 --exfactor 2.0
"""
import os
import argparse
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi

import bigfish.stack as stack
# Use the 3D versions you provided
from bigfish.classification.features3d import (
    features_distance3d,
    features_in_out_nucleus3d,
    features_dispersion3d,
    features_topography3d,
    features_area3d,
    get_features_name3d,
    opening_by_edt,
)
from bigfish.classification.input_preparation3d import _get_centroid_surface3d, _get_centroid_rna3d

# ----------------------------- Paths and defaults -----------------------------
strFn_CIDs = r"Y:\Users\XSJ\WK_XSJ\Pool_Results\tbCIDs.csv"
strFn_Genes = r"Y:\Users\XSJ\WK_XSJ\Pool_Results\clGenes.csv"
strDir_NeucliMasks = r"Y:\Users\XSJ\WK_XSJ\Pool_Results\Neucli_Masks_Clean"
strDir_CellMasks  = r"Y:\Users\XSJ\WK_XSJ\Pool_Results\RNALocs_Mask_Crop"
strDir_Sav       = r"Y:\Users\XSJ\WK_XSJ\Pool_Results\RNALocs_Features_Results5"


# ----------------------------- Round-level cache builder -----------------------------
def _precompute_round_cache(cell_mask, nuc_mask, voxel_size_zyx):
    """Precompute RNA-independent derived quantities for a single round (same CID/R)."""
    vz, vy, vx = map(float, voxel_size_zyx)
    sampling = (vz, vy, vx)

    # 1) Distance to cell membrane (nm)
    distance_cell = ndi.distance_transform_edt(cell_mask, sampling=sampling).astype(np.float32)

    # 2) Cell centroid (geometric)
    centroid_cell = _get_centroid_surface3d(cell_mask).astype(np.int64)

    # 3) Nucleus-related
    if nuc_mask is not None:
        nuc_mask = nuc_mask.astype(bool)
        cell_mask_out_nuc = cell_mask.copy()
        cell_mask_out_nuc[nuc_mask] = False

        # Distance outside nucleus: consistent with `prepare_extracted_data3d`
        dist_nuc_out = ndi.distance_transform_edt(~nuc_mask, sampling=sampling).astype(np.float32)
        distance_nuc = (cell_mask * dist_nuc_out).astype(np.float32)

        centroid_nuc = _get_centroid_surface3d(nuc_mask).astype(np.int64)
    else:
        cell_mask_out_nuc = None
        distance_nuc = np.zeros_like(distance_cell, dtype=np.float32)
        centroid_nuc = np.array([0, 0, 0], dtype=np.int64)

    # 4) Protrusion opening (3D EDT variant), then mark nuclear voxels as solid to avoid misclassification
    mask_opened = opening_by_edt(cell_mask, r_nm=3000.0, voxel_size_zyx=voxel_size_zyx)
    if nuc_mask is not None:
        mask_opened[nuc_mask] = True

    # 5) Area/volume block (gene-independent, reusable; repeated per gene for column alignment)
    area_features = features_area3d(cell_mask, nuc_mask if nuc_mask is not None else np.zeros_like(cell_mask, bool),
                                    cell_mask_out_nuc if cell_mask_out_nuc is not None else np.zeros_like(cell_mask, bool),
                                    voxel_size_zyx, check_input=False)

    return {
        "distance_cell": distance_cell,
        "distance_nuc": distance_nuc,
        "centroid_cell": centroid_cell,
        "centroid_nuc": centroid_nuc,
        "cell_mask_out_nuc": cell_mask_out_nuc if cell_mask_out_nuc is not None else np.zeros_like(cell_mask, bool),
        "mask_opened": mask_opened,
        "area_features": area_features,
    }


def _features_protrusion3d_cached(rna_coord, cell_mask, nuc_mask, voxel_size_zyx, mask_opened):
    """Equivalent to `features_protrusion3d` but reuses the precomputed `mask_opened`."""
    nb_rna = len(rna_coord)
    cell_volume = int(cell_mask.sum())
    vox_vol = np.prod(voxel_size_zyx) / 1e9  # nm^3 -> µm^3

    # Upstream code ensures: mask_opened[nuc_mask] = True
    protrusion_volume = (cell_volume - int(mask_opened.sum())) * vox_vol  # µm^3

    if nb_rna == 0:
        return (1.0, 0.0, float(protrusion_volume))

    z, y, x = rna_coord[:, 0], rna_coord[:, 1], rna_coord[:, 2]
    kept = mask_opened[z, y, x]  # True means still in the "non-protrusion" region
    nb_rna_protrusion = int(nb_rna - np.count_nonzero(kept))
    expected_rna_protrusion = nb_rna * (protrusion_volume / float(cell_volume)) if cell_volume > 0 else 0.0
    index_rna_protrusion = (nb_rna_protrusion / expected_rna_protrusion) if expected_rna_protrusion > 0 else 1.0
    proportion_rna_protrusion = (nb_rna_protrusion / float(nb_rna)) if nb_rna > 0 else 0.0
    return (float(index_rna_protrusion), float(proportion_rna_protrusion), float(protrusion_volume))


# ----------------------------- Per-cell processing with round cache -----------------------------
def process_one_cell_with_cache(strCID: str, clGenes, nRC: int, fExFactor: float) -> pd.DataFrame:
    """Process a single cell (CID), building per-round caches that are reused across six genes."""
    dfGeneF_List = []

    for nR in range(1, nRC + 1):
        strFn_NeuclMask = os.path.join(strDir_NeucliMasks, f"{strCID}_R{nR}_Neucl.tif")
        strFn_CellMask  = os.path.join(strDir_CellMasks,  f"{strCID}", f"{strCID}_R{nR}_Mask.tif")
        strFn_VoxSz     = os.path.join(strDir_CellMasks,  f"{strCID}", f"{strCID}_R{nR}_VoxSz.csv")

        # Existence checks
        bNeucli = os.path.exists(strFn_NeuclMask)
        bCell   = os.path.exists(strFn_CellMask)
        bVoxSz  = os.path.exists(strFn_VoxSz)

        if not (bNeucli and bCell and bVoxSz):
            if not bNeucli:
                print(f"{strFn_NeuclMask} does not exist!")
            if not bCell:
                print(f"{strFn_CellMask} does not exist!")
            if not bVoxSz:
                print(f"{strFn_VoxSz} does not exist!")
            continue

        try:
            imgCellMask  = stack.read_image(strFn_CellMask).astype(bool)
            imgNeuclMask = stack.read_image(strFn_NeuclMask).astype(bool)
            dfVox = pd.read_csv(strFn_VoxSz, delimiter=",")
            voxel_size_zyx = tuple((dfVox.loc[0, ["Z", "Y", "X"]].to_numpy() * 1000.0 / fExFactor).astype(float))  # nm
        except Exception as e:
            print(f"[{strCID} R{nR}] I/O error: {e}")
            continue

        # Round-level precomputation
        cache = _precompute_round_cache(imgCellMask, imgNeuclMask, voxel_size_zyx)

        # Feature name list, aligned with `compute_features3d`
        feature_names = get_features_name3d(
            names_features_distance=True,
            names_features_intranuclear=True,
            names_features_protrusion=True,
            names_features_dispersion=True,
            names_features_topography=True,
            names_features_foci=False,
            names_features_area=True,
            names_features_centrosome=False,
        )

        # Area features (tuple), reused per gene
        area_tuple = cache["area_features"]

        # Per-gene
        idxG_S = (nR - 1) * 6
        for nG in range(6):
            idxG = idxG_S + nG
            if idxG >= len(clGenes):
                continue

            strG = clGenes[idxG]
            strFn_RNALocs = os.path.join(strDir_CellMasks, f"{strCID}", f"{strCID}_R{nR}_{strG}.csv")
            if not os.path.exists(strFn_RNALocs):
                print(f"{strFn_RNALocs} does not exist!")
                continue

            try:
                dfRNALocs = pd.read_csv(strFn_RNALocs, delimiter=",")
                matRNALocs = np.round(dfRNALocs.loc[:, ["Z", "Y", "X"]].to_numpy()).astype(np.int64) - 1  # 1->0
                matRNALocs = np.clip(matRNALocs, [0, 0, 0], np.array(imgCellMask.shape) - 1)
            except Exception as e:
                print(f"[{strCID} R{nR} {strG}] RNA I/O error: {e}")
                continue

            # RNA outside nucleus
            if matRNALocs.size > 0:
                z, y, x = matRNALocs[:, 0], matRNALocs[:, 1], matRNALocs[:, 2]
                in_nuc = imgNeuclMask[z, y, x]
                rna_out = matRNALocs[~in_nuc] if in_nuc.size > 0 else matRNALocs
            else:
                rna_out = np.empty((0, 3), dtype=np.int64)

            # Concatenate feature blocks in the same order as `compute_features3d`
            feats = []

            # 1) distance
            feats += list(features_distance3d(matRNALocs, cache["distance_cell"], cache["distance_nuc"], imgCellMask, check_input=False))

            # 2) intranuclear
            feats += list(features_in_out_nucleus3d(matRNALocs, rna_out, check_input=False))

            # 3) protrusion (reuse precomputed mask_opened)
            feats += list(_features_protrusion3d_cached(matRNALocs, imgCellMask, imgNeuclMask, voxel_size_zyx, cache["mask_opened"]))

            # 4) dispersion
            centroid_rna = _get_centroid_rna3d(matRNALocs) if len(matRNALocs) > 0 else np.array([0,0,0], dtype=np.int64)
            feats += list(features_dispersion3d(None, matRNALocs, centroid_rna, imgCellMask, cache["centroid_cell"], cache["centroid_nuc"], check_input=False))

            # 5) topography
            feats += list(features_topography3d(matRNALocs, imgCellMask, imgNeuclMask, cache["cell_mask_out_nuc"], voxel_size_zyx, check_input=False))

            # 6) foci (not enabled)

            # 7) area (gene-independent; repeated for column alignment)
            feats += list(area_tuple)

            # 8) centrosome (not enabled)

            # Assemble into a DataFrame row
            arr = np.array(feats, dtype=np.float32).reshape(1, -1)
            dfGeneF = pd.DataFrame(data=arr, columns=feature_names)
            dfGeneF.insert(0, "Gene", strG)
            dfGeneF.insert(0, "CID", f"{strCID}")
            dfGeneF_List.append(dfGeneF)

    if len(dfGeneF_List) == 0:
        return pd.DataFrame()

    dfCellF = pd.concat(dfGeneF_List, ignore_index=True)
    os.makedirs(strDir_Sav, exist_ok=True)
    strFn_CellF_Sav = os.path.join(strDir_Sav, f"{strCID}_RNALocF.csv")
    try:
        dfCellF.to_csv(strFn_CellF_Sav, index=False)
    except Exception as e:
        print(f"[{strCID}] save error: {e}")
    return dfCellF


# ----------------------------- Entry point (parallel) -----------------------------
def main(n_jobs: int, nRC: int, fExFactor: float):
    # Load tables
    dfCIDs  = pd.read_csv(strFn_CIDs)
    dfGenes = pd.read_csv(strFn_Genes, header=None)
    clGenes = dfGenes.iloc[0].tolist()

    cids = [f"{row.ANMID}_{row.NID}" for _, row in dfCIDs.iterrows()]
    print(f"Total cells (CIDs): {len(cids)}")

    # Parallel execution (loky backend)
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(process_one_cell_with_cache)(cid, clGenes, nRC, fExFactor) for cid in cids
    )

    results = [df for df in results if isinstance(df, pd.DataFrame) and not df.empty]
    if len(results) == 0:
        print("No features generated.")
        return

    dfAll = pd.concat(results, ignore_index=True)
    os.makedirs(strDir_Sav, exist_ok=True)
    strFn_DatasetF_Sav = os.path.join(strDir_Sav, "Dataset_RNALocF.csv")
    dfAll.to_csv(strFn_DatasetF_Sav, index=False)
    print(f"Saved dataset: {strFn_DatasetF_Sav}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="Number of parallel workers (default: -1)")
    parser.add_argument("--rc", type=int, default=5, help="Number of rounds per cell (default: 5)")
    parser.add_argument("--exfactor", type=float, default=2.0, help="External scale factor to recover true voxel size from microns (default: 2.0)")
    args = parser.parse_args()

    main(n_jobs=args.jobs, nRC=args.rc, fExFactor=args.exfactor)
