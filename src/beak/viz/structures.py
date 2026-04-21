"""Visualization functions for protein structures"""

import os
import urllib
import urllib.request
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

from ..structures.mapping import align_sequences


ALPHAFOLD_PDB_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"


def three_to_one(resname):
    """Convert 3-letter residue name to 1-letter code (X if unknown)."""
    return protein_letters_3to1.get(resname.capitalize(), "X")


def save_mapped_structure(input, value_dict, reference_seq, null_bfactor=0.0):
    """
    Map desired position-wise values to the B-factor column of a PDB file,
    aligning the reference sequence to the PDB-derived sequence.

    Parameters
    ----------
    input : str
        Path to the input PDB file, PDB ID, or UniProt Accession Number.
    value_dict : dict
        Dict with keys as reference sequence positions (1-based) and values (float).
    reference_seq : str
        Reference amino acid sequence (1-letter code).
    null_bfactor : float, optional
        Default B-factor value for residues not in value_dict (default=0.0).

    Returns
    -------
    str
        Path to output PDB with mapped B-factors.
    """

    # --- Fetch structure if needed ---
    if ".pdb" not in input:
        if len(input) == 4:  # PDB ID
            pdb_id = input
            urllib.request.urlretrieve(
                f"http://files.rcsb.org/download/{pdb_id}.pdb", f"{pdb_id}.pdb"
            )
            pdb_filepath = f"{pdb_id}.pdb"
        elif os.path.isdir(input):
            print("dir! add support for directories...") # ADD DIR SUPPORT
            os.mkdir('mapped_structures')
        else:  # UniProt Accession → AlphaFold
            pdb_filepath = f"{input}_AF.pdb"
            urllib.request.urlretrieve(
                ALPHAFOLD_PDB_URL.format(uniprot_id=input), pdb_filepath
            )
    else:
        pdb_filepath = input

    # --- Extract sequence from PDB ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_filepath)
    model = next(structure.get_models())
    chain = next(model.get_chains())  # assume single chain
    pdb_residues = [res for res in chain.get_residues() if res.id[0] == " "]
    pdb_seq = "".join(three_to_one(res.get_resname()) for res in pdb_residues)

    # --- Align reference sequence to PDB sequence ---
    ref_aln, pdb_aln = align_sequences(reference_seq, pdb_seq)

    # --- Build mapping (ref_pos → pdb_residue) ---
    mapping = {}
    ref_pos, pdb_pos = 0, 0
    for r_char, p_char in zip(ref_aln, pdb_aln):
        if r_char != "-":
            ref_pos += 1
        if p_char != "-":
            pdb_pos += 1
        if r_char != "-" and p_char != "-":
            mapping[ref_pos] = pdb_residues[pdb_pos - 1]

    # --- Modify PDB file lines ---
    with open(pdb_filepath, "r") as pdb_file:
        pdb_lines = pdb_file.readlines()

    modified_pdb_lines = []
    for line in pdb_lines:
        if line.startswith("ATOM"):
            residue_index = int(line[22:26].strip())

            # set to null value first
            b_factor = null_bfactor

            # overwrite if mapping has a value
            for ref_pos, residue in mapping.items():
                if residue.get_id()[1] == residue_index:
                    if ref_pos in value_dict:
                        b_factor = value_dict[ref_pos]
                    break

            line = line[:60] + f"{b_factor:6.2f}" + line[66:]
            modified_pdb_lines.append(line)
        else:
            modified_pdb_lines.append(line)

    # --- Save modified file ---
    output_filepath = f"mapped_value_{os.path.basename(pdb_filepath)}"
    with open(output_filepath, "w") as f:
        f.writelines(modified_pdb_lines)

    print(f"Modified structure saved to {output_filepath}")
    return output_filepath
