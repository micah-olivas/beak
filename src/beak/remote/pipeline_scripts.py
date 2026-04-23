"""Bash-script generators for each pipeline step type.

Pure string builders — no SSH, no state. The `Pipeline` class in
`pipeline.py` dispatches to these by step_type.
"""

from typing import List


def generate_search_commands(step, step_dir, input_file, output_name, remote_path,
                             db_base_path) -> List[str]:
    """Generate MMseqs2 search commands."""
    from .search import MMseqsSearch

    database = step.params.get('database', 'uniref90')

    if database in MMseqsSearch.AVAILABLE_DBS:
        db_file = MMseqsSearch.AVAILABLE_DBS[database]
        db_path = f"{db_base_path}/{db_file}"
    elif database.startswith('/'):
        db_path = database
    else:
        db_path = f"{db_base_path}/{database}"

    # Handle preset - expand to actual parameters
    params_to_use = step.params.copy()

    if 'preset' in params_to_use:
        preset_name = params_to_use.pop('preset')
        if preset_name in MMseqsSearch.PRESETS:
            preset_params = MMseqsSearch.PRESETS[preset_name]['params'].copy()
            for k, v in preset_params.items():
                if k not in params_to_use:
                    params_to_use[k] = v

    param_str = []
    for k, v in params_to_use.items():
        if k == 'database':
            continue
        param_name = k.replace("_", "-")
        prefix = "-" if len(k) == 1 else "--"
        param_str.append(f'{prefix}{param_name} {v}')
    param_str = ' '.join(param_str)

    return [
        "# Create query database",
        f"mmseqs createdb {remote_path}/{input_file} {step_dir}/queryDB --dbtype 1",
        "",
        "# Run MMseqs2 search",
        f"mmseqs search \\",
        f"  {step_dir}/queryDB \\",
        f"  {db_path} \\",
        f"  {step_dir}/resultDB \\",
        f"  {step_dir}/tmp \\",
        f"  {param_str}",
        "",
        "# Convert to m8 format",
        f"mmseqs convertalis \\",
        f"  {step_dir}/queryDB \\",
        f"  {db_path} \\",
        f"  {step_dir}/resultDB \\",
        f"  {step_dir}/results.m8",
        "",
        "# Extract hit sequences",
        f"mmseqs createseqfiledb {db_path} {step_dir}/resultDB {step_dir}/seqDB",
        f"mmseqs result2flat {db_path} {db_path} {step_dir}/seqDB {step_dir}/{output_name}.fasta --use-fasta-header",
        "",
        "# Count hits",
        f"CONTEXT[hit_count]=$(grep -c '^>' {step_dir}/{output_name}.fasta || echo 0)",
        f"echo \"Found ${{CONTEXT[hit_count]}} hits\"",
        "",
        "# Cleanup",
        f"rm -rf {step_dir}/tmp {step_dir}/queryDB* {step_dir}/resultDB* {step_dir}/seqDB*"
    ]


def generate_filter_commands(step, step_dir, input_file, remote_path) -> List[str]:
    """Generate sequence filtering commands using Python/BioPython."""
    input_fasta = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file

    script = ["from Bio import SeqIO", "import re", ""]
    script.append(f"records = list(SeqIO.parse('{input_fasta}', 'fasta'))")

    if step.params.get('size'):
        min_len, max_len = step.params['size']
        script.append(f"records = [r for r in records if {min_len} <= len(r.seq) <= {max_len}]")

    if step.params.get('motif'):
        motif = step.params['motif']
        script.append(f"records = [r for r in records if re.search(r'{motif}', str(r.seq))]")

    if step.params.get('remove_fragments'):
        script.append("records = [r for r in records if (str(r.seq).count('-') + str(r.seq).count('X')) / len(r.seq) <= 0.1]")

    if step.params.get('deduplicate'):
        script.append("seen = set()")
        script.append("unique = [r for r in records if not (str(r.seq) in seen or seen.add(str(r.seq)))]")
        script.append("records = unique")

    if step.params.get('max_sequences'):
        n = step.params['max_sequences']
        script.append(f"records = sorted(records, key=lambda x: len(x.seq), reverse=True)[:{n}]")

    script.append(f"SeqIO.write(records, '{step_dir}/filtered.fasta', 'fasta')")
    script.append(f"print(f'Filtered to {{len(records)}} sequences')")

    return [
        f"cat > {step_dir}/filter.py << 'EOF'\n" + '\n'.join(script) + "\nEOF",
        f"python3 {step_dir}/filter.py",
        f"CONTEXT[filtered_count]=$(grep -c '^>' {step_dir}/filtered.fasta || echo 0)"
    ]


def generate_taxonomy_commands(step, step_dir, input_file, remote_path,
                               db_base_path) -> List[str]:
    """Generate MMseqs2 taxonomy commands."""
    from .taxonomy import MMseqsTaxonomy

    database = step.params.get('database', 'uniprotkb')
    db_path = (f"{db_base_path}/{MMseqsTaxonomy.AVAILABLE_DBS[database]}"
            if database in MMseqsTaxonomy.AVAILABLE_DBS
            else database if database.startswith('/')
            else f"{db_base_path}/{database}")

    params = ['--tax-lineage 1'] if step.params.get('tax_lineage', True) else []
    params.extend(f"{'--' if len(k) > 1 else '-'}{k.replace('_', '-')} {v}"
                for k, v in step.params.items() if k not in ['database', 'tax_lineage'])

    input_fasta = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file

    return [
        f"mmseqs createdb {input_fasta} {step_dir}/queryDB",
        f"mmseqs taxonomy {step_dir}/queryDB {db_path} {step_dir}/taxResult {step_dir}/tmp {' '.join(params)}",
        f"mmseqs createtsv {step_dir}/queryDB {step_dir}/taxResult {step_dir}/taxonomy.tsv",
        f"rm -rf {step_dir}/tmp {step_dir}/queryDB* {step_dir}/taxResult*"
    ]


def generate_align_commands(step, step_dir, input_file, remote_path) -> List[str]:
    """Generate alignment commands for clustalo, mafft, or muscle."""
    from .align import _CMD_BUILDERS, ALGORITHMS

    algorithm = step.params.pop('algorithm', 'clustalo')
    output_format = step.params.pop('output_format', 'fasta')

    algo_info = ALGORITHMS.get(algorithm, ALGORITHMS['clustalo'])
    log_path = f"{step_dir}/{algo_info['log_file']}"
    output_path = f"{step_dir}/alignment.{output_format}"
    filtered = f"{step_dir}/filtered_input.fasta"

    input_fasta = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file

    align_cmd = _CMD_BUILDERS[algorithm](
        filtered, output_path, output_format, log_path, step.params
    )

    return [
        "# Filter empty sequences",
        f"if command -v seqkit &> /dev/null; then",
        f"  seqkit seq -m 1 {input_fasta} > {filtered}",
        f"else",
        f"  awk '/^>/ {{if (seq) print header \"\\n\" seq; header=$0; seq=\"\"; next}} {{seq=seq $0}} END {{if (seq) print header \"\\n\" seq}}' {input_fasta} | \\",
        f"    awk 'BEGIN {{RS=\">\"; ORS=\"\"}} NR>1 {{split($0,a,\"\\n\"); seq=\"\"; for(i=2;i<=length(a);i++) seq=seq a[i]; if (length(seq)>0) print \">\" $0}}' > {filtered}",
        f"fi",
        "",
        f"echo \"Sequences for alignment: $(grep -c '^>' {filtered} || echo 0)\"",
        "",
        f"# Run {algo_info['name']}",
        align_cmd
    ]


def generate_tree_commands(step, step_dir, input_file, remote_path) -> List[str]:
    """Generate IQ-TREE commands with a safe fallback output."""
    input_alignment = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file

    params = []
    for key, value in step.params.items():
        param_name = key.replace("_", "-")
        prefix = "-" if len(key) == 1 else "--"
        params.append(f"{prefix}{param_name} {value}")

    return [
        f"if command -v iqtree2 &> /dev/null; then",
        f"  iqtree2 -s {input_alignment} -pre {step_dir}/tree {' '.join(params)}",
        f"elif command -v iqtree &> /dev/null; then",
        f"  iqtree -s {input_alignment} -pre {step_dir}/tree {' '.join(params)}",
        f"else",
        f"  echo '(iqtree not available)' > {step_dir}/tree.nwk",
        f"fi",
        f"if [ -f {step_dir}/tree.treefile ]; then cp {step_dir}/tree.treefile {step_dir}/tree.nwk; fi"
    ]


def generate_embeddings_commands(step, step_dir, input_file, remote_path,
                                 remote_job_dir,
                                 docker_dir=None,
                                 project_name='beak') -> List[str]:
    """Generate embedding commands using Docker service.

    docker_dir defaults to `{remote_job_dir}/docker` for backwards
    compatibility; pass an absolute path (e.g. from Pipeline._resolve_docker_dir)
    to target a shared service. project_name is threaded through so the
    docker compose invocation scopes to the same project that deployed the
    service.
    """
    if docker_dir is None:
        docker_dir = f"{remote_job_dir}/docker"

    model = step.params.get('model', 'esm2_t33_650M_UR50D')
    input_fasta = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file
    repr_layers = step.params.get('repr_layers', [-1])
    repr_layers_str = ' '.join(str(layer) for layer in repr_layers)
    embedding_flags = []
    if step.params.get('include_mean', True):
        embedding_flags.append('--include-mean')
    if step.params.get('include_per_tok', False):
        embedding_flags.append('--include-per-tok')
    flags_str = ' '.join(embedding_flags)
    gpu_id = int(step.params.get('gpu_id', 0))

    return [
        f"# Generate embeddings using Docker service",
        f"cd {docker_dir}",
        f"mkdir -p {step_dir}/embeddings",
        f"docker compose --project-name {project_name} exec -T embeddings "
        f"python /app/generate_embeddings.py "
        f"--input {input_fasta} "
        f"--output {step_dir}/embeddings "
        f"--model {model} "
        f"--repr-layers {repr_layers_str} "
        f"{flags_str} "
        f"--gpu {gpu_id}"
    ]
