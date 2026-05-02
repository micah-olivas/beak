"""Functions to load bundled reference datasets."""

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

GROWTH_TEMPS_FILE = "enqvist_growth_temps.tsv"
GROWTH_TEMPS_URL = "https://zenodo.org/records/1175609/files/temperature_data.tsv?download=1"

# metaTraits — EMBL 2026, ~140 harmonised organism traits across 2.2M genomes,
# from BacDive + BV-BRC + JGI IMG + GOLD with NCBI/GTDB tax mapping.
# Distribution: https://metatraits.embl.de  (CC-BY-SA, no auth).
#
# Downloads are taxonomy-level JSONL.gz summaries, separated by rank
# (species / genus / family) and by tax authority (NCBI / GTDB). Our
# `taxonomy.parquet` carries NCBI taxon_ids and species-level scientific
# names, so we default to the NCBI species file and fall through to genus
# locally inside `annotate_traits`.
_METATRAITS_BASE = "https://metatraits.embl.de/static/downloads"
METATRAITS_RANKS = ("species", "genus", "family")
METATRAITS_TAXONOMIES = ("ncbi", "gtdb")


def download_growth_temps(force: bool = False) -> Path:
    """Download the Enqvist growth-temperature dataset from Zenodo.

    Args:
        force: Re-download even if the file already exists locally.

    Returns:
        Path to the local TSV file.

    Reference:
        Enqvist, M. et al. (2018). Growth temperatures of prokaryotes.
        Zenodo. https://doi.org/10.5281/zenodo.1175609
    """
    dest = DATA_DIR / GROWTH_TEMPS_FILE

    if dest.exists() and not force:
        return dest

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading growth-temperature dataset (~5 MB) from Zenodo...")
    urllib.request.urlretrieve(GROWTH_TEMPS_URL, dest)
    print(f"✓ Saved to {dest}")
    return dest


def load_growth_temps() -> pd.DataFrame:
    """Load the Enqvist microbial growth-temperature dataset.

    Auto-downloads from Zenodo on first call (~5 MB, cached locally).

    Returns:
        DataFrame with columns:
            - organism: organism name (lowercased, underscore-joined)
            - domain: Bacteria / Archaea / Eukaryota
            - temperature: optimal growth temperature (°C)
            - taxid: NCBI taxonomy ID
            - lineage_text: full lineage string
            - superkingdom, phylum, class, order, family, genus: taxid per level

    Reference:
        Enqvist, M. et al. (2018). Growth temperatures of prokaryotes.
        Zenodo. https://doi.org/10.5281/zenodo.1175609

    Example:
        >>> temps = load_growth_temps()
        >>> temps[temps['organism'].str.contains('escherichia')]
    """
    data_file = download_growth_temps()
    return pd.read_csv(data_file, sep='\t')


def metatraits_url(rank: str = "species", taxonomy: str = "ncbi") -> str:
    """Resolve the metaTraits download URL for a (rank, taxonomy) pair.

    $BEAK_METATRAITS_URL takes precedence so a user can point at a local
    mirror or a dated snapshot regardless of rank.
    """
    override = os.environ.get("BEAK_METATRAITS_URL")
    if override:
        return override
    if rank not in METATRAITS_RANKS:
        raise ValueError(f"rank must be one of {METATRAITS_RANKS}, got {rank!r}")
    if taxonomy not in METATRAITS_TAXONOMIES:
        raise ValueError(
            f"taxonomy must be one of {METATRAITS_TAXONOMIES}, got {taxonomy!r}"
        )
    return f"{_METATRAITS_BASE}/{taxonomy}_{rank}_summary.jsonl.gz"


def _metatraits_cache_path(rank: str, taxonomy: str) -> Path:
    return DATA_DIR / f"metatraits_{taxonomy}_{rank}.parquet"


_NO_MAJORITY = "No robust majority"
_RE_BOOL = __import__("re").compile(r"^(true|false):\s*\((\d+)%\)\s*$", __import__("re").I)
_RE_LABEL = __import__("re").compile(r"^(.+?):\s*\((\d+)%\)\s*$")
_RE_MEDIAN = __import__("re").compile(r"^Median:\s*(-?\d+(?:\.\d+)?)")
_RE_SLUG = __import__("re").compile(r"[^a-z0-9]+")


def _parse_majority_label(label, is_discrete: bool):
    """Convert a metaTraits ``majority_label`` string into a primitive.

    Discrete labels look like ``"true: (88%)"`` or ``"rod-shaped: (76%)"``.
    Continuous labels look like ``"Median: 30.9 Celsius"``. Anything
    ambiguous (``"No robust majority"`` or unparseable) returns None so
    downstream code treats it as missing rather than as a literal string.
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s or s == _NO_MAJORITY:
        return None
    if is_discrete:
        m = _RE_BOOL.match(s)
        if m:
            return m.group(1).lower() == "true"
        m = _RE_LABEL.match(s)
        if m:
            return m.group(1).strip()
        return s
    m = _RE_MEDIAN.match(s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _slug(name: str) -> str:
    """Normalise a trait name into a snake_case identifier."""
    return _RE_SLUG.sub("_", name.strip().lower()).strip("_")


def _jsonl_gz_to_dataframe(path: Path) -> pd.DataFrame:
    """Parse a metaTraits JSONL.gz into a wide trait DataFrame.

    Output schema: one row per organism, columns are ``scientific_name``
    plus one column per trait (snake_cased). Bool / float / str cell
    values come from the upstream ``majority_label`` parser.
    """
    import gzip

    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            flat = {"scientific_name": rec.get("tax_name")}
            summaries = rec.get("summaries") or []
            for item in summaries:
                name = item.get("name")
                if not name:
                    continue
                val = _parse_majority_label(
                    item.get("majority_label"),
                    bool(item.get("is_discrete", True)),
                )
                if val is None:
                    continue
                flat[_slug(name)] = val
            rows.append(flat)
    return pd.DataFrame(rows)


def download_metatraits(
    force: bool = False,
    rank: str = "species",
    taxonomy: str = "ncbi",
    url: Optional[str] = None,
) -> Optional[Path]:
    """Download the metaTraits taxonomy-level summary and cache as parquet.

    First call streams the source JSONL.gz, parses it, and writes a
    parquet next to the bundled datasets so subsequent loads are fast.

    Args:
        force: Re-download + re-parse even if the cached parquet exists.
        rank: One of "species" (default), "genus", "family".
        taxonomy: One of "ncbi" (default) or "gtdb".
        url: Override the resolved download URL.

    Returns:
        Path to the cached parquet, or None on failure. Non-fatal by
        design — callers should treat traits as optional metadata.

    Reference:
        metaTraits, EMBL 2026. https://metatraits.embl.de
    """
    cache = _metatraits_cache_path(rank, taxonomy)
    if cache.exists() and not force:
        return cache

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    src = url or metatraits_url(rank=rank, taxonomy=taxonomy)
    tmp_jsonl = DATA_DIR / f"metatraits_{taxonomy}_{rank}.jsonl.gz"
    try:
        print(f"Downloading metaTraits ({taxonomy}/{rank}) from {src}...")
        urllib.request.urlretrieve(src, tmp_jsonl)
        df = _jsonl_gz_to_dataframe(tmp_jsonl)
        if df.empty:
            print("metaTraits download returned no rows.")
            return None
        # Normalise the join key — upstream uses `taxonomy_id` for NCBI
        # files but our taxonomy.parquet writes `taxon_id`.
        if "taxon_id" not in df.columns and "taxonomy_id" in df.columns:
            df = df.rename(columns={"taxonomy_id": "taxon_id"})
        # Some traits carry both booleans and multi-class strings across
        # rows (e.g. `presence_of_hemolysis` mixes true/false with "beta").
        # Stringify any mixed-type column so parquet can write it.
        for col in df.columns:
            if col == "scientific_name":
                continue
            if df[col].dtype == object:
                kinds = {type(v).__name__ for v in df[col].dropna()}
                if len(kinds) > 1:
                    df[col] = df[col].map(
                        lambda v: None if v is None
                        else (str(v).lower() if isinstance(v, bool) else str(v))
                    )
        df.to_parquet(cache, index=False)
        print(f"✓ Cached {len(df):,} rows to {cache}")
        return cache
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"metaTraits download failed: {e}")
        return None
    finally:
        # Drop the gzip; parquet is the canonical local form.
        if tmp_jsonl.exists():
            try:
                tmp_jsonl.unlink()
            except OSError:
                pass


def load_metatraits(
    rank: str = "species",
    taxonomy: str = "ncbi",
) -> Optional[pd.DataFrame]:
    """Load the metaTraits taxonomy-level summary.

    Auto-downloads + caches on first call. Returns None when the dataset
    isn't reachable.

    Args:
        rank: One of "species" (default), "genus", "family".
        taxonomy: One of "ncbi" (default) or "gtdb".

    Schema (from EMBL JSONL summaries):
        - ``taxon_id`` (NCBI; renamed from upstream ``taxonomy_id``) or
          GTDB taxon identifier
        - ``scientific_name``, lineage levels (domain through species)
        - ~140 harmonised trait columns: optimum_ph, oxygen_requirement,
          motility, sporulation, gram_stain, salinity_optimum,
          temperature_optimum, gc_content, doubling_time_hours,
          isolation_source, cell_shape, ...

    Reference:
        metaTraits, EMBL 2026. https://metatraits.embl.de
    """
    path = download_metatraits(rank=rank, taxonomy=taxonomy)
    if path is None or not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def list_datasets() -> pd.DataFrame:
    """List all available bundled datasets."""
    metadata_file = DATA_DIR / "metadata.json"

    if not metadata_file.exists():
        return pd.DataFrame([
            {
                'name': 'enqvist_growth_temps',
                'description': 'Microbial growth temperatures',
                'source': 'https://zenodo.org/records/1175609',
                'loader': 'load_growth_temps()'
            },
            {
                'name': 'metatraits',
                'description': '140+ harmonised organism traits across 2.2M genomes',
                'source': 'https://metatraits.embl.de',
                'loader': 'load_metatraits()'
            },
        ])

    with open(metadata_file) as f:
        meta = json.load(f)

    return pd.DataFrame(meta['datasets'])
