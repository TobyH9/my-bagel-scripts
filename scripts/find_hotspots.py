import biotite.structure as bs
from biotite.structure import AtomArray
from pathlib import Path
import numpy as np
from typing import cast
import numpy.typing as npt
from biotite.structure.io import load_structure as load_structure_biotite
import biotite.sequence as seq
from biotite.sequence import ProteinSequence
from scipy.spatial.distance import pdist, squareform
import fire

def load_structure(
    filepath: Path,
    filter_non_empty_ins_code: bool = True,
    filter_hetero: bool = True,
) -> bs.AtomArray:
    """
    Load a macromolecular structure from a local CIF file.

    Optionally filters out:
      - residues with any non-empty insertion code
      - heteroatom residues (non-standard residues, ligands, etc.)
      - residues missing any backbone atom (N, CA, or C)

    Args:
        filepath: Path to a `.cif` file.
        filter_non_empty_ins_code: Remove residues that have any non-empty insertion code.
        filter_hetero: Remove heteroatom residues.

    Returns:
        Biotite AtomArray containing filtered atomic coordinates.
    """
    structure = load_structure_biotite(str(filepath))

    mask = np.ones(len(structure), dtype=bool)

    # --- Filter heteroatoms ---
    if filter_hetero:
        mask &= ~structure.hetero

    # --- Filter residues with non-empty insertion codes ---
    if filter_non_empty_ins_code:
        bad_res_keys = np.unique(
            [
                f"{c}:{r}"
                for c, r, ins in zip(structure.chain_id, structure.res_id, structure.ins_code)
                if ins != ""
            ]
        )
        res_keys = np.array([f"{c}:{r}" for c, r in zip(structure.chain_id, structure.res_id)])
        mask &= ~np.isin(res_keys, bad_res_keys)
    else:
        res_keys = np.array([f"{c}:{r}" for c, r in zip(structure.chain_id, structure.res_id)])

    # --- Filter residues missing any backbone atom (N, CA, C) ---
    valid_res = []
    for key in np.unique(res_keys[mask]):
        atoms = structure.atom_name[mask & (res_keys == key)]
        if {"N", "CA", "C"}.issubset(atoms):
            valid_res.append(key)
    mask &= np.isin(res_keys, valid_res)

    return structure[mask]

def extract_sequence_and_residue_coordinates(
    atom_array: AtomArray,
    chain_id: str,
    start_res: int | None = None,
    end_res: int | None = None,
) -> tuple[str, list[npt.NDArray[np.float64]]]:
    """Extract sequence and per-residue atom coordinate arrays for a domain.

    Extracts protein sequence and atomic coordinates for a specified domain region
    within a protein chain. Filters by chain ID and residue range, then processes
    each residue to extract CA atoms and convert residue names to one-letter codes.

    Args:
        atom_array (structure.AtomArray): Biotite AtomArray of the structure.
        chain_id (str): Chain identifier character.
        start_res (int | None): Starting residue ID (inclusive). Defaults to None.
        end_res (int | None): Ending residue ID (inclusive). Defaults to None.

    Returns:
        Tuple[str, list[np.ndarray]]: Domain sequence and residue atom coordinates.
            - Domain sequence (str): One-letter amino acid sequence.
            - Residue atoms (list[np.ndarray]): list of coordinate arrays.
                Expected shape for each array: (N_atoms_in_residue, 3)
    """
    mask_res = atom_array.chain_id == chain_id
    if start_res is not None:
        mask_res &= atom_array.res_id >= start_res
    if end_res is not None:
        mask_res &= atom_array.res_id <= end_res
    subset = atom_array[mask_res]
    if subset.array_length() == 0:
        return "", []

    mask_ca = subset.atom_name == "CA"
    ca_subset = subset[mask_ca]
    if ca_subset.array_length() == 0:
        return "", []

    residue_ids: list[int] = list(map(int, ca_subset.res_id.tolist()))

    seq_chars: list[str] = []
    residue_atoms: list[npt.NDArray[np.float64]] = []
    for res_id in residue_ids:
        res_mask = subset.res_id == res_id
        res_atoms = subset[res_mask]
        if res_atoms.array_length() == 0:
            continue
        ca_mask = res_atoms.atom_name == "CA"
        if np.any(ca_mask):
            res_name = res_atoms[ca_mask][0].res_name
        else:
            res_name = res_atoms[0].res_name

        try:
            seq_char = ProteinSequence.convert_letter_3to1(res_name)
        except Exception:
            continue

        coords = res_atoms.coord.astype(float)
        if coords.ndim != 2 or coords.shape[1] != 3:
            continue

        seq_chars.append(seq_char)
        residue_atoms.append(coords)

    sequence = "".join(seq_chars)
    return sequence, residue_atoms

def get_cb_coord(
    C: npt.NDArray[np.float64],
    N: npt.NDArray[np.float64],
    CA: npt.NDArray[np.float64],
    length: float = 1.522,
    angle: float = 1.927,
    dihedral: float = -2.143,
) -> npt.NDArray[np.float64]:
    """
    Get the Cβ coordinate from the C, N, and CA coordinates.
    Args:
        C (npt.NDArray[np.float64]): A numpy array of C coordinates (in Angstroms).
        N (npt.NDArray[np.float64]): A numpy array of N coordinates (in Angstroms).
        CA (npt.NDArray[np.float64]): A numpy array of CA coordinates (in Angstroms).
        length (float): Length (in Angstroms). Defaults to 1.522.
        angle (float): Angle (in radians). Defaults to 1.927.
        dihedral (float): Dihedral (in radians). Defaults to -2.143.
    Returns:
        Cβ coordinate (in Angstroms) (npt.NDArray[np.float64])
    """

    def normalize(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return cast(npt.NDArray[np.float64], x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True))

    bc = normalize(N - CA)
    n = normalize(np.cross(N - C, bc))
    m = [bc, np.cross(n, bc), n]
    d = [
        length * np.cos(angle),
        length * np.sin(angle) * np.cos(dihedral),
        -length * np.sin(angle) * np.sin(dihedral),
    ]
    return cast(npt.NDArray[np.float64], CA + sum([m * d for m, d in zip(m, d)]))

def compute_contact_map(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: str | None = None,
) -> npt.NDArray[np.int64]:
    """
    Compute the contact map for a structure.
    Args:
        structure: Biotite ``AtomArray`` containing the structure.
        distance_threshold: Distance threshold (in Angstroms).
        chain: Chain identifier. If not specified, all chains are considered.
    Returns:
        Contact map (numpy array)
            - Expected shape: (L, L), where L is the total number of residues across all chains.
            - Values in {-1, 0, 1}.
                -1: Missing/invalid distance (e.g., NaN)
                 0: No contact (distance >= threshold)
                 1: Contact (distance < threshold)
    """
    mask = np.ones(len(structure), dtype=bool)
    if chain is not None:
        mask &= structure.chain_id == chain

    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]

    Cbeta = get_cb_coord(C=C, N=N, CA=CA)
    dist = squareform(pdist(Cbeta))

    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return cast(npt.NDArray[np.int64], contacts)

def main(file: str,
         target_chain_id: str,
         receptor_chain_id: str,
         threshold: float) -> list[list[int]]:

    cif = Path(file)
    structure = load_structure(cif)

    virus_seq, _ = extract_sequence_and_residue_coordinates(structure, chain_id=target_chain_id)
    receptor_seq, _ = extract_sequence_and_residue_coordinates(structure, chain_id=receptor_chain_id)
    virus_len = len(virus_seq)
    receptor_len = len(receptor_seq)

    contact_map = compute_contact_map(structure, distance_threshold=threshold)

    interchain_contacts = contact_map[0:(virus_len), virus_len:(virus_len+receptor_len)]

    binding = interchain_contacts > 0
    virus_contacts = np.argwhere(binding)
    virus_contacts = np.unique(virus_contacts[:, 0])

    virus_residues_contacts = np.sort(virus_contacts)

    # Find breaks where consecutive numbers stop
    split_indices = np.where(np.diff(virus_residues_contacts) != 1)[0] + 1

    # Split into subarrays
    groups = np.split(virus_residues_contacts, split_indices)
    # Extend the groups by two residues
    extended_groups = [np.arange(g[0] - 1, g[-1] + 2).tolist() for g in groups]

    return extended_groups

if __name__ == "__main__":
    fire.Fire(main)
