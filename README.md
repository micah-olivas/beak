<div style="display: flex; align-items: top;">
     <img src="misc/icon.png" alt="BEAK icon" align="right" height=200pt/>
     <h1 style="margin: 0;">BEAK</h1>
</div>

**B**iophysical and **E**volutionary **A**nalysis **K**it

A Python toolkit for interpreting protein experiments in evolutionary and structural context.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

BEAK assembles a target-centric view of one protein: biochemical measurements (activity, stability, binding) joined to conservation, taxonomy, growth temperature, and structural features (pLDDT, secondary structure, burial, contacts) at every residue, keyed on target-sequence position. The result is a single table for filtering, joining, and plotting.

The expensive parts (MMseqs2 searches, Clustal Omega alignments, HMMER scans, ESM embeddings) run remotely over SSH. BEAK submits those jobs, tracks them locally, and pulls results back into the interpretation layer. That remote compute is infrastructure; the product is the joined data at a single protein's residues.

## Installation

```bash
git clone https://github.com/micah-olivas/beak.git
cd beak
pip install -e ".[dev]"     # drop [dev] for a runtime-only install
```

### Requirements

- Python 3.8+
- SSH access to a remote server with bioinformatics tools installed (for remote steps only; analysis, import, and export work offline)
- Core dependencies (`pandas`, `biopython`, `fabric`, `click`, `rich`, `textual`, `torch`, ...) install automatically

## Remote Setup

Point BEAK at your server, then verify the connection:

```bash
beak config init     # interactive: host, user, SSH key, remote job dir → ~/.beak/config.toml
beak doctor          # report reachable tools, databases, and disk usage on the remote
```

BEAK auto-detects an SSH key at `~/.ssh/id_ed25519` or `~/.ssh/id_rsa`. If your lab already runs a shared BEAK server, `beak config init` + `beak doctor` is all you need.

Tools expected on the remote:

| Tool                                                    | Used by                                  |
| ------------------------------------------------------- | ---------------------------------------- |
| [MMseqs2](https://github.com/soedinglab/MMseqs2)        | sequence search, LCA taxonomy            |
| [Clustal Omega](http://www.clustal.org/omega/) / MAFFT / MUSCLE | multiple sequence alignment      |
| [IQ-TREE](http://www.iqtree.org/)                       | phylogenetic trees                       |
| [HMMER3](http://hmmer.org/)                             | Pfam domain scans (`beak pfam`)          |

Two optional remote setups:

```bash
beak setup pfam              # download + index Pfam-A HMM DB (~5 GB) for `beak pfam`
beak setup pfam --system     # shared, world-readable install (needs sudo on the remote)
```

ESM embeddings run in a Docker container on the remote. Each user deploys their own by default; to share one service across a lab, point everyone at a common directory with `beak config set docker.service_dir /srv/beak_docker` (users must be in the `docker` group).

## Package Structure

```
src/beak/
├── cli/            # Click + Rich CLI, one module per command group
├── project/        # BeakProject: the on-disk, target-centric analysis hub
├── remote/         # Fabric SSH job managers (search, align, embeddings,
│                   #   taxonomy, hmmer, pipeline) + BeakSession
├── api/            # REST clients (UniProt, PDBe SIFTS, AlphaFold)
├── alignments/     # alignment cache (FASTA→npz), conservation, PSSM formatting
├── analysis/       # comparative PSSMs (group / differential)
├── structures/     # per-residue pLDDT/SASA/contacts/SS, alignment↔PDB mapping
├── viz/            # PSSM logos, ChimeraX export
├── tui/            # Textual TUI (browse projects, layers, alignments, taxonomy)
├── embeddings.py   # embedding loaders + PCA cache
├── temperature.py  # organism → growth-temperature annotation (Enqvist dataset)
├── sequence.py     # FASTA parsing, sequence properties
├── config.py       # ~/.config/beak/config.toml read/write
└── datasets/       # auto-downloaded reference data
```

## Usage

### Projects

A project is a target-centric analysis hub on disk at `~/.beak/projects/<name>/`. It anchors on one protein sequence and accretes layers (homolog sets, taxonomy, structure, conservation), each keyed back to the target's residue positions.

```bash
beak project init my_kinase --uniprot P00533     # from a UniProt accession
beak project init my_kinase --sequence target.fasta   # or a local FASTA
beak project list                                  # all projects, with status + size
beak project status my_kinase                      # which layers are populated / stale
beak ui                                            # browse and build projects in the TUI
```

`beak ui` opens the Textual interface, where most project work happens: adding homolog sets, inspecting alignments and taxonomy, running comparative analyses, and submitting remote jobs.

### Remote building blocks

`BeakSession` is the recommended entry point for scripting remote jobs: one SSH connection fanned out to per-job-type managers, all reading defaults from `~/.beak/config.toml`.

```python
from beak.remote import BeakSession

bk = BeakSession()                                 # or BeakSession(host=..., user=...)

job = bk.search.submit("query.fasta", database="uniref90", e=0.001, threads=8)
bk.search.wait(job)
hits = bk.search.get_results(job)                  # pandas DataFrame

# Pfam domain scan (synchronous)
for hit in bk.hmmer.scan("query.fasta"):
    print(hit["pfam_id"], hit["pfam_name"], hit["i_evalue"])
```

Individual managers (`MMseqsSearch`, `MMseqsTaxonomy`, `ClustalAlign`, `ESMEmbeddings`, `HmmerScan`, `Pipeline`) are also importable directly from `beak.remote`.

### Pipelines

Chain search → taxonomy → align → tree into a single remote job:

```python
from beak.remote import Pipeline

pipe = Pipeline(host="your-server.edu", user="your_username")
pipe.search("query.fasta", database="uniref90", e=0.001) \
    .taxonomy(database="uniref90") \
    .align(threads=4) \
    .tree(bootstrap=1000)

job_id = pipe.execute(job_name="my_analysis")
pipe.print_detailed_status(job_id)                 # per-step progress
alignment = pipe.get_step_results(job_id, step_number=3)
```

Steps can be made conditional on hit counts:

```python
pipe.search("query.fasta", database="uniref90") \
    .if_min_hits(10).then("taxonomy", database="uniref90") \
    .if_max_hits(1000).then("align") \
    .tree()
```

### Command-line jobs

Every remote step is also a top-level command. Jobs run asynchronously; state is tracked in `~/.beak/jobs.json`.

```bash
beak search query.fasta --db uniref90 --preset broad   # submit a search
beak jobs                                               # list jobs, auto-refresh
beak status <job_id> --watch                            # follow one job
beak log <job_id> --follow                              # tail its remote log
beak results <job_id> --parse                           # pull results as a DataFrame

beak pfam query.fasta --uniprot --taxonomy              # scan domains, fetch UniProt hits
beak structures P00533 --source alphafold               # download structures
beak features model.cif --format parquet                # per-residue structural features
```

## Citation

If you use BEAK in your research, please cite:

```
[Citation information coming soon]
```

## License

MIT License. See [LICENSE](LICENSE).

## Contributing

Contributions welcome. Please open an issue or submit a pull request.
