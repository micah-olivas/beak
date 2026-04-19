"""Per-residue structural feature extraction from mmCIF files."""

import functools
import math
import shutil
import warnings

import gemmi
import pandas as pd


def extract_structure_features(
    cif_path: str,
    chain_id: str = None,
    skip_dssp: bool = False,
    skip_sasa: bool = False,
) -> pd.DataFrame:
    """Extract per-residue structural features from an mmCIF file.

    Parameters
    ----------
    cif_path : str
        Path to an mmCIF (.cif) file.
    chain_id : str, optional
        Chain to extract. If None, uses the first polymer chain.
    skip_dssp : bool
        If True, skip secondary structure assignment entirely.
    skip_sasa : bool
        If True, skip SASA computation.

    Returns
    -------
    pd.DataFrame
        One row per resolved residue. Columns: residue_number (int),
        residue_name (str, 1-letter), plddt (float), secondary_structure
        (str: H/E/C), sasa (float), n_contacts (int), burial_wcn (float).
    """
    structure, chain, chain_name = _parse_chain(cif_path, chain_id)
    polymer = chain.get_polymer()

    # Base residue info
    residue_numbers = []
    residue_names = []
    for res in polymer:
        info = gemmi.find_tabulated_residue(res.name)
        one_letter = info.one_letter_code if info.one_letter_code != '?' else 'X'
        residue_numbers.append(res.seqid.num)
        residue_names.append(one_letter)

    # pLDDT / B-factor
    plddt = _extract_plddt(polymer)

    # Contacts and WCN
    contacts, wcn = _compute_contacts_and_wcn(polymer, structure)

    # Secondary structure
    if skip_dssp:
        ss = {rn: 'C' for rn in residue_numbers}
    else:
        ss = _assign_secondary_structure(cif_path, chain_name)

    # SASA
    if skip_sasa:
        sasa = {rn: float('nan') for rn in residue_numbers}
    else:
        sasa = _compute_sasa(cif_path, chain_name)

    # Assemble DataFrame
    rows = []
    for resnum, resname in zip(residue_numbers, residue_names):
        rows.append({
            'residue_number': resnum,
            'residue_name': resname,
            'plddt': plddt.get(resnum, float('nan')),
            'secondary_structure': ss.get(resnum, 'C'),
            'sasa': sasa.get(resnum, float('nan')),
            'n_contacts': contacts.get(resnum, 0),
            'burial_wcn': wcn.get(resnum, 0.0),
        })

    return pd.DataFrame(rows)


def _parse_chain(cif_path, chain_id=None):
    """Parse mmCIF and return (structure, chain, chain_name).

    Raises ValueError if chain not found.
    """
    structure = gemmi.read_structure(cif_path)
    model = structure[0]

    if chain_id is not None:
        chain = model.find_chain(chain_id)
        if chain is None:
            raise ValueError(
                f"Chain '{chain_id}' not found in {cif_path}. "
                f"Available chains: {[c.name for c in model]}"
            )
        return structure, chain, chain_id
    else:
        for c in model:
            polymer = c.get_polymer()
            if len(polymer) > 0:
                return structure, c, c.name
        raise ValueError(f"No polymer chain found in {cif_path}")


def _extract_plddt(polymer):
    """Extract B-factor from CA atoms (pLDDT for AlphaFold structures).

    Returns dict of {residue_number: float}.
    """
    plddt = {}
    for res in polymer:
        ca = res.find_atom("CA", '*')
        if ca is not None:
            plddt[res.seqid.num] = ca.b_iso
    return plddt


def _get_representative_atom(residue):
    """Get CB atom, or CA for glycine. Returns None if neither found."""
    if residue.name == 'GLY':
        return residue.find_atom("CA", '*')
    cb = residue.find_atom("CB", '*')
    if cb is not None:
        return cb
    return residue.find_atom("CA", '*')


def _compute_contacts_and_wcn(polymer, structure, contact_radius=8.0, min_seq_separation=4):
    """Compute inter-residue contacts and weighted contact number.

    Contacts: CB-CB distance < contact_radius, sequence separation >= min_seq_separation.
    WCN: sum(1/r^2) for all representative atom pairs.

    Returns ({resnum: n_contacts}, {resnum: wcn}).
    """
    # Collect representative atoms with their sequence index and residue number
    rep_atoms = []
    for seq_idx, res in enumerate(polymer):
        atom = _get_representative_atom(res)
        if atom is not None:
            rep_atoms.append((seq_idx, res.seqid.num, atom.pos))

    contacts = {}
    wcn = {}

    for i, (seq_i, resnum_i, pos_i) in enumerate(rep_atoms):
        n_contacts = 0
        wcn_sum = 0.0

        for j, (seq_j, resnum_j, pos_j) in enumerate(rep_atoms):
            if i == j:
                continue

            dist = pos_i.dist(pos_j)
            if dist < 0.01:
                continue

            # WCN uses all pairs
            wcn_sum += 1.0 / (dist * dist)

            # Contacts require sequence separation and distance cutoff
            if abs(seq_i - seq_j) >= min_seq_separation and dist < contact_radius:
                n_contacts += 1

        contacts[resnum_i] = n_contacts
        wcn[resnum_i] = wcn_sum

    return contacts, wcn


@functools.lru_cache(maxsize=1)
def _dssp_available():
    """Check if mkdssp binary is available on the system."""
    return shutil.which("mkdssp") is not None


def _assign_secondary_structure(cif_path, chain_id):
    """Assign secondary structure: H (helix), E (strand), C (coil).

    Uses DSSP if mkdssp is available, otherwise falls back to parsing
    _struct_conf and _struct_sheet_range records from the mmCIF.

    Returns dict of {residue_number: 'H'|'E'|'C'}.
    """
    if _dssp_available():
        return _compute_dssp(cif_path, chain_id)
    else:
        warnings.warn(
            "mkdssp not found; using mmCIF secondary structure annotations. "
            "Install mkdssp for DSSP-based assignment.",
            stacklevel=2,
        )
        return _parse_ss_from_mmcif(cif_path, chain_id)


def _compute_dssp(cif_path, chain_id):
    """Run DSSP via BioPython wrapper.

    Returns dict of {residue_number: 'H'|'E'|'C'}.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.DSSP import DSSP

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)
    model = structure[0]

    dssp = DSSP(model, cif_path, dssp="mkdssp")

    ss_map = {}
    for key in dssp.keys():
        dssp_chain, dssp_resid = key
        if dssp_chain != chain_id:
            continue
        resnum = dssp_resid[1]
        ss_code = dssp[key][2]
        # Map DSSP codes to simplified H/E/C
        if ss_code in ('H', 'G', 'I'):
            ss_map[resnum] = 'H'
        elif ss_code in ('E', 'B'):
            ss_map[resnum] = 'E'
        else:
            ss_map[resnum] = 'C'

    return ss_map


def _parse_ss_from_mmcif(cif_path, chain_id):
    """Parse secondary structure from mmCIF _struct_conf and _struct_sheet_range.

    Returns dict of {residue_number: 'H'|'E'|'C'}.
    """
    doc = gemmi.cif.read(cif_path)
    block = doc.sole_block()
    ss_map = {}

    # Parse helices from _struct_conf
    struct_conf = block.find(
        '_struct_conf.',
        ['conf_type_id', 'beg_auth_asym_id', 'beg_auth_seq_id',
         'end_auth_asym_id', 'end_auth_seq_id']
    )
    for row in struct_conf:
        conf_type = row[0]
        beg_chain = row[1]
        end_chain = row[3]

        if not conf_type.startswith('HELX'):
            continue
        if beg_chain != chain_id:
            continue

        try:
            beg_res = int(row[2])
            end_res = int(row[4])
            for resnum in range(beg_res, end_res + 1):
                ss_map[resnum] = 'H'
        except (ValueError, TypeError):
            continue

    # Parse sheets from _struct_sheet_range
    sheet_range = block.find(
        '_struct_sheet_range.',
        ['beg_auth_asym_id', 'beg_auth_seq_id',
         'end_auth_asym_id', 'end_auth_seq_id']
    )
    for row in sheet_range:
        beg_chain = row[0]

        if beg_chain != chain_id:
            continue

        try:
            beg_res = int(row[1])
            end_res = int(row[3])
            for resnum in range(beg_res, end_res + 1):
                ss_map[resnum] = 'E'
        except (ValueError, TypeError):
            continue

    return ss_map


def _compute_sasa(cif_path, chain_id):
    """Compute per-residue SASA using BioPython's ShrakeRupley.

    Returns dict of {residue_number: float}.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser
    from Bio.PDB.SASA import ShrakeRupley

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("model", cif_path)
    model = structure[0]

    sr = ShrakeRupley()
    sr.compute(structure, level='R')

    sasa_map = {}
    for chain in model:
        if chain.id != chain_id:
            continue
        for residue in chain:
            if residue.id[0] != ' ':
                continue
            resnum = residue.id[1]
            sasa_map[resnum] = residue.sasa

    return sasa_map
