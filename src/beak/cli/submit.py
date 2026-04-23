"""Job submission commands: search, taxonomy, align, embeddings."""

import click
from pathlib import Path

from .main import main
from ._common import get_manager, auto_name_from_pfam


# Known ESM2 embedding dimensions — used for upfront size estimation.
# Unknown models (custom finetunes etc.) fall back to skipping the estimate.
_ESM2_EMBED_DIM = {
    'esm2_t6_8M_UR50D': 320,
    'esm2_t12_35M_UR50D': 480,
    'esm2_t30_150M_UR50D': 640,
    'esm2_t33_650M_UR50D': 1280,
}


def _humanize_bytes(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024 or unit == 'TB':
            return f"{n:.1f} {unit}"
        n /= 1024


def _fasta_stats(path) -> tuple:
    """Scan a FASTA file without loading it into memory.

    Returns (n_seqs, total_length, max_length, min_length).
    Raises click.BadParameter if the file has no records or all seqs empty.
    """
    n_seqs = 0
    total_len = 0
    max_len = 0
    min_len = None
    current_len = 0

    def _flush():
        nonlocal total_len, max_len, min_len
        if current_len == 0:
            return
        total_len += current_len
        max_len = max(max_len, current_len)
        min_len = current_len if min_len is None else min(min_len, current_len)

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                _flush()
                n_seqs += 1
                current_len = 0
            else:
                # ignore blank lines between records
                current_len += len(line)
        _flush()

    if n_seqs == 0:
        raise click.BadParameter(
            f"{path!r} has no FASTA records (no '>' headers found).",
            param_hint="'INPUT_FILE'",
        )
    if total_len == 0:
        raise click.BadParameter(
            f"{path!r} has {n_seqs} headers but zero sequence content.",
            param_hint="'INPUT_FILE'",
        )
    return n_seqs, total_len, max_len, (min_len or 0)


def _estimate_embedding_bytes(n_seqs: int, total_len: int, model: str,
                              n_layers: int, include_mean: bool,
                              include_per_tok: bool) -> int:
    """Estimate the uncompressed pickle size (pre .npz compression) in bytes.

    Returns None for unknown model dimensions.
    """
    dim = _ESM2_EMBED_DIM.get(model)
    if dim is None:
        return None
    bytes_mean = n_seqs * dim * n_layers * 4 if include_mean else 0
    bytes_tok = total_len * dim * n_layers * 4 if include_per_tok else 0
    return bytes_mean + bytes_tok


# Warn + confirm threshold when output crosses this many bytes.
_EMBEDDING_SIZE_WARN_BYTES = 5 * 1024 ** 3  # 5 GB


@main.command()
@click.argument('query', required=True)
@click.option('--db', 'database', required=True, help='Database alias (e.g., uniref90)')
@click.option('--name', 'job_name', default=None, help='Job name')
@click.option('--preset', default=None,
              type=click.Choice(['default', 'fast', 'sensitive', 'exhaustive', 'very_sensitive']),
              help='Search preset')
@click.option('--uniprot', is_flag=True,
              help='Treat QUERY as a UniProt accession ID instead of a file path')
def search(query, database, job_name, preset, uniprot):
    """Submit an MMseqs2 search job.

    QUERY is a path to a FASTA file, or a UniProt accession if --uniprot is set.
    """
    if uniprot:
        from ..api.uniprot import fetch_uniprot
        query_file = fetch_uniprot(query)
        if job_name is None:
            job_name = f"{query}_search"
    else:
        query_file = query
        if not Path(query_file).exists():
            raise click.BadParameter(f"File not found: {query_file}", param_hint="'QUERY'")

    if job_name is None:
        job_name = auto_name_from_pfam(query_file, 'search')

    mgr = get_manager(job_type='search')
    kwargs = {}
    if preset:
        kwargs['preset'] = preset
    mgr.submit(query_file, database=database, job_name=job_name, **kwargs)


@main.command()
@click.argument('query', required=True)
@click.option('--db', 'database', default='uniprotkb', help='Database alias')
@click.option('--name', 'job_name', default=None, help='Job name')
@click.option('--no-lineage', is_flag=True, help='Skip lineage parsing')
@click.option('--uniprot', is_flag=True,
              help='Treat QUERY as a UniProt accession ID instead of a file path')
def taxonomy(query, database, job_name, no_lineage, uniprot):
    """Submit an MMseqs2 taxonomy job.

    QUERY is a path to a FASTA file, or a UniProt accession if --uniprot is set.
    """
    if uniprot:
        from ..api.uniprot import fetch_uniprot
        query_file = fetch_uniprot(query)
        if job_name is None:
            job_name = f"{query}_taxonomy"
    else:
        query_file = query
        if not Path(query_file).exists():
            raise click.BadParameter(f"File not found: {query_file}", param_hint="'QUERY'")

    if job_name is None:
        job_name = auto_name_from_pfam(query_file, 'taxonomy')

    mgr = get_manager(job_type='taxonomy')
    mgr.submit(query_file, database=database, job_name=job_name,
               tax_lineage=not no_lineage)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--name', 'job_name', default=None, help='Job name')
@click.option('--algorithm', '-a', default='clustalo',
              type=click.Choice(['clustalo', 'mafft', 'muscle']),
              help='Alignment algorithm (default: clustalo)')
@click.option('--format', 'output_format', default=None,
              help='Output format (default depends on algorithm)')
def align(input_file, job_name, algorithm, output_format):
    """Submit a multiple sequence alignment job"""
    if job_name is None:
        job_name = auto_name_from_pfam(input_file, 'align')

    mgr = get_manager(job_type='align')
    kwargs = {'algorithm': algorithm}
    if output_format:
        kwargs['output_format'] = output_format
    mgr.submit(input_file, job_name=job_name, **kwargs)


@main.command()
@click.argument('input_file', type=click.Path(exists=True), required=False)
@click.option('--from-job', 'source_job_id', default=None,
              help='Chain off a completed search job: use its hits.fasta as '
                   'the input, copied directly on the remote (no local '
                   'round-trip). Mutually exclusive with INPUT_FILE.')
@click.option('--model', '-m', default='esm2_t33_650M_UR50D',
              help='ESM model ID (default: esm2_t33_650M_UR50D; see --list-models)')
@click.option('--name', 'job_name', default=None, help='Job name')
@click.option('--layer', 'repr_layers', default='-1',
              help='Comma-separated representation layers to extract (default: -1)')
@click.option('--per-tok', 'include_per_tok', is_flag=True,
              help='Also save per-token embeddings (large files)')
@click.option('--no-mean', is_flag=True,
              help='Skip mean-pooled embeddings (only makes sense with --per-tok)')
@click.option('--gpu', 'gpu_id', default=0, help='GPU device ID on the remote')
@click.option('--list-models', is_flag=True,
              help='List available ESM models and exit')
def embeddings(input_file, source_job_id, model, job_name, repr_layers,
               include_per_tok, no_mean, gpu_id, list_models):
    """Submit an ESM embedding-generation job.

    INPUT_FILE is a local FASTA file. Alternatively, use --from-job to
    feed in the hits FASTA of a completed search job without downloading
    it to your machine.

    Results come back as pickles; load them with
    `beak.embeddings.load_mean_embeddings()` or via `mgr.get_results(job_id)`.

    Examples:

        beak embeddings library.fasta -m esm2_t12_35M_UR50D

        beak embeddings --from-job <search_id> -m esm2_t12_35M_UR50D

        beak embeddings library.fasta --per-tok --no-mean

        beak embeddings library.fasta --layer 30,33 --per-tok

        beak embeddings --list-models
    """
    from ..remote.embeddings import ESMEmbeddings

    if list_models:
        click.echo(ESMEmbeddings.list_models().to_string(index=False))
        return

    if bool(input_file) == bool(source_job_id):
        raise click.UsageError(
            "Provide exactly one of INPUT_FILE or --from-job <job_id>."
        )

    try:
        layers = [int(x.strip()) for x in repr_layers.split(',')]
    except ValueError:
        raise click.BadParameter(
            f"--layer must be a comma-separated list of ints, got {repr_layers!r}",
            param_hint="'--layer'",
        )

    include_mean = not no_mean
    if not include_mean and not include_per_tok:
        raise click.UsageError(
            "Nothing to output: --no-mean requires --per-tok."
        )

    mgr = get_manager(job_type='embeddings')

    if source_job_id:
        remote_input = _resolve_source_job_fasta(source_job_id)
        if job_name is None:
            job_name = f"embeddings_from_{source_job_id}"
        mgr.submit(
            remote_input=remote_input,
            model=model,
            job_name=job_name,
            repr_layers=layers,
            include_mean=include_mean,
            include_per_tok=include_per_tok,
            gpu_id=gpu_id,
        )
        return

    # Local-FASTA path: validate structure and show a size estimate so
    # the user doesn't kick off a silent multi-hundred-GB job on --per-tok.
    n_seqs, total_len, max_len, min_len = _fasta_stats(input_file)
    avg_len = total_len // n_seqs
    click.echo(
        f"Input: {n_seqs:,} sequences "
        f"(avg {avg_len} aa, min {min_len}, max {max_len})"
    )

    est_bytes = _estimate_embedding_bytes(
        n_seqs, total_len, model, len(layers), include_mean, include_per_tok,
    )
    if est_bytes is not None:
        tag = ""
        if include_per_tok and include_mean:
            tag = " (mean + per-token)"
        elif include_per_tok:
            tag = " (per-token only)"
        click.echo(f"Estimated output size: ~{_humanize_bytes(est_bytes)}{tag}")
        if est_bytes > _EMBEDDING_SIZE_WARN_BYTES:
            click.confirm(
                f"  ⚠  Output will exceed {_humanize_bytes(_EMBEDDING_SIZE_WARN_BYTES)}. Continue?",
                abort=True, default=False,
            )

    if job_name is None:
        job_name = auto_name_from_pfam(input_file, 'embeddings')

    mgr.submit(
        input_file=input_file,
        model=model,
        job_name=job_name,
        repr_layers=layers,
        include_mean=include_mean,
        include_per_tok=include_per_tok,
        gpu_id=gpu_id,
    )


def _resolve_source_job_fasta(job_id: str) -> str:
    """Resolve a source job's output FASTA to an absolute remote path.

    Supported source job types:
      - search: ensures hits.fasta exists on the remote (runs mmseqs
        createseqfiledb/result2flat if needed) and returns its path.
      - align:  returns alignment.fasta (produced in-place by the
        alignment step).

    Raises click.BadParameter with an actionable message on unsupported
    types or missing jobs.
    """
    import json
    from pathlib import Path

    db_path = Path.home() / ".beak" / "jobs.json"
    if not db_path.exists():
        raise click.BadParameter(
            f"No job database at {db_path}. Has anything been submitted?",
            param_hint="'--from-job'",
        )
    with open(db_path) as f:
        job_db = json.load(f)
    if job_id not in job_db:
        raise click.BadParameter(
            f"Unknown job id '{job_id}' (not in {db_path}).",
            param_hint="'--from-job'",
        )

    info = job_db[job_id]
    jtype = info.get('job_type')

    if jtype == 'search':
        from ..remote.search import MMseqsSearch
        src_mgr = MMseqsSearch()
        return src_mgr.ensure_remote_hits_fasta(job_id)

    if jtype == 'align':
        remote_path = info['remote_path']
        # Default output name for align is alignment.fasta; if the user
        # picked a non-fasta output format this will be wrong and the
        # Docker run will fail with a clear FileNotFoundError.
        return f"{remote_path}/alignment.fasta"

    raise click.BadParameter(
        f"Can't chain embeddings from job type '{jtype}'. "
        f"Supported: search, align.",
        param_hint="'--from-job'",
    )
