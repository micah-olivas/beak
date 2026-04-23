"""Job submission commands: search, taxonomy, align."""

import click
from pathlib import Path

from .main import main
from ._common import get_manager, auto_name_from_pfam


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
