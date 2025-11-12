<div style="display: flex; align-items: top;">
     <img src="misc/icon-transparent.png" alt="BEAK icon" align="right" height=200pt/>
     <h1 style="margin: 0;">BEAK</h1>
</div>

**B**iophysical and **E**volutionary **A**ssociations **K**it

A Python toolkit for remote bioinformatics workflows, designed for experimental biophysicists and biochemists. BEAK orchestrates computationally intensive sequence analysis tasks on a remote server while providing a seamless notebook-based interface.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Features

### Remote Job Orchestration
- **Pipeline Builder**: Chain multiple analysis steps into automated workflows
- **Smart Job Management**: Track, monitor, and retrieve results seamlessly
- **Real-time Progress**: Detailed step-by-step status tracking

### Sequence Search
- **[MMseqs2](https://github.com/soedinglab/MMseqs2) Integration**: Large-scale sequence searches against major databases
- **Hit Extraction**: Automatically retrieve hit sequences as FASTA files
- **Taxonomy Assignment**: Annotate taxonomies with MMseqs2 taxonomy

### Alignment & Phylogenetics
- **[Clustal Omega](http://www.clustal.org/omega/)**: Multiple sequence alignment
- **[IQ-TREE](http://www.iqtree.org/)**: Maximum likelihood phylogenetic trees
- **Pipeline Integration**: Seamlessly chain search → filter → align → tree workflows

### Structure & Embeddings (Coming Soon)
- **[ColabFold](https://github.com/sokrypton/ColabFold)**: AlphaFold2 structure prediction
- **[ESM](https://github.com/facebookresearch/esm)** & **[ProGen](https://github.com/salesforce/progen)**: Protein language model embeddings

### Analysis Ready
- Parse results directly into pandas DataFrames
- Export sequences, alignments, and trees in standard formats
- Cache all parameters for full reproducibility

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/beak.git
cd beak
pip install -e .
```

### Remote Server Setup

BEAK requires SSH access to a remote server with bioinformatics tools installed.

**1. Generate SSH key (if you don't have one):**

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

**2. Copy your public key to the remote server:**

```bash
ssh-copy-id username@your-server.edu
```

**3. Test your connection:**

```bash
ssh username@your-server.edu
```

BEAK will automatically detect your SSH key at `~/.ssh/id_ed25519` or `~/.ssh/id_rsa`. To use a custom key, specify `key_path` when initializing.

**Required tools on remote server:**
- [MMseqs2](https://github.com/soedinglab/MMseqs2) for sequence search and taxonomy
- [Clustal Omega](http://www.clustal.org/omega/) for alignment
- [IQ-TREE](http://www.iqtree.org/) for phylogenetics

---

## Usage Examples

### Simple Sequence Search

```python
from beak.remote import MMseqsSearch

# Initialize connection (auto-detects SSH key)
search = MMseqsSearch(
    host="your-server.edu",
    user="your_username"
)

# List available databases
search.list_databases()

# Submit a search job
job_id = search.submit(
    query_file="my_sequences.fasta",
    database="uniref90",
    job_name="my_search",
    e=0.001,
    threads=8
)

# Check status
search.status(job_id)

# Get results as a DataFrame
results = search.get_results(job_id)

# Extract hit sequences
hits_fasta = search.get_hit_sequences(job_id)
```

### Automated Pipeline

```python
from beak.remote import Pipeline

# Create pipeline
pipe = Pipeline(
    host="your-server.edu",
    user="your_username"
)

# Define workflow
pipe.search("query.fasta", database="uniref90", e=0.001, threads=8) \
    .taxonomy(database="uniref90") \
    .align(threads=4) \
    .tree(bootstrap=1000)

# View pipeline
print(pipe)
# Output:
# Pipeline:
#   Input: query.fasta
#   Steps (4):
#     1. search (database=uniref90, e=0.001, threads=8)
#     2. taxonomy (database=uniref90)
#     3. align (threads=4)
#     4. tree (bootstrap=1000)

# Execute
job_id = pipe.execute(job_name="my_analysis")

# Monitor progress
pipe.print_detailed_status(job_id)
# Output:
# ============================================================
# Pipeline: my_analysis (abc12345)
# Status: RUNNING | Runtime: 0:05:23
# ============================================================
#   ✓ Step 1: search (database=uniref90, e=0.001, threads=8) [COMPLETED]
#     └─ hits: 164
#   ⟳ Step 2: taxonomy (database=uniref90) [RUNNING]
#   ○ Step 3: align (threads=4) [PENDING]
#   ○ Step 4: tree (bootstrap=1000) [PENDING]
# ============================================================

# Download results from specific steps
search_results = pipe.get_step_results(job_id, step_number=1)
alignment = pipe.get_step_results(job_id, step_number=3)
```

### Conditional Pipelines

```python
# Only align if we find between 10-1000 hits
pipe.search("query.fasta", database="uniref90", e=0.001) \
    .if_min_hits(10).then('taxonomy', database="uniref90") \
    .if_max_hits(1000).then('align') \
    .tree()

job_id = pipe.execute()
```

### Taxonomy Assignment

```python
from beak.remote import MMseqsTaxonomy

# Initialize
taxonomy = MMseqsTaxonomy(
    host="your-server.edu",
    user="your_username"
)

# Assign taxonomy with full lineages
job_id = taxonomy.submit(
    query_file="sequences.fasta",
    database="uniref90",
    tax_lineage=True
)

# Get taxonomic annotations
tax_results = taxonomy.get_results(job_id)
# Returns DataFrame with columns: query, taxid, rank, scientific_name, lineage
```

---

## Motivation

BEAK addresses common challenges in modeling evolutionary sequence relationships:

1. **Computational Resources**: Large-scale sequence searches and alignments are too intensive for personal laptops
2. **Workflow Integration**: Difficult to integrate remote computational steps into exploratory data analysis
3. **Reproducibility**: Parameters and workflows need to be traceable and comparable across projects
4. **Pipeline Complexity**: Chaining multiple tools requires manual scripting and file management

**Solution**: BEAK works within your local notebook environment while seamlessly offloading compute-intensive tasks to remote servers. All job parameters are automatically cached, and pipelines can be defined declaratively, making analyses traceable, reproducible, and easy to iterate.

---

## Requirements

- Python 3.8+
- SSH access to a remote server with bioinformatics tools installed
- Dependencies: `fabric`, `pandas`, `biopython`, `paramiko`

---

## Documentation

[Coming soon]

---

## Citation

If you use BEAK in your research, please cite:

```
[Citation information coming soon]
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

---

## Contact

micah5051olivas@gmail.com
