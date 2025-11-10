Here's a polished README with the new remote MMseqs2 functionality highlighted:

```markdown
<div style="display: flex; align-items: top;">
     <img src="misc/icon-transparent.png" alt="BEAK icon" align="right" height=200pt/>
     <h1 style="margin: 0;">BEAK</h1>
</div>

**B**iophysical **E**volutionary **A**nalysis **K**it

A Python toolkit for evolutionary sequence analysis, designed for experimental biophysicists and biochemists.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Features

### 🔍 Remote Sequence Search
- **MMseqs2 Integration**: Submit large-scale sequence searches to remote servers directly from your notebook
- **Database Support**: UniRef90, UniRef100, SwissProt, TrEMBL, and more
- **Job Management**: Track, monitor, and retrieve results seamlessly
- **Automatic Setup**: Auto-detects SSH keys and configures remote directories

### 🧬 Taxonomy Assignment
- Taxonomically label sequences using MMseqs2 taxonomy
- Full lineage support for phylogenetic context
- Integrated with major protein databases

### 📊 Analysis Ready
- Parse results directly into pandas DataFrames
- Extract hit sequences as FASTA files
- Cache search parameters for reproducibility

---

## Quick Start

### Installation

```bash
pip install beak
```

Or install from source:

```bash
git clone https://github.com/yourusername/beak.git
cd beak
pip install -e .
```

### Remote Sequence Search

```python
from beak.new_remote import MMseqsSearch

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

### Taxonomy Assignment

```python
from beak.new_remote import MMseqsTaxonomy

# Initialize
taxonomy = MMseqsTaxonomy(
    host="your-server.edu",
    user="your_username"
)

# Assign taxonomy
job_id = taxonomy.submit(
    query_file="sequences.fasta",
    database="uniref90",
    tax_lineage=True
)

# Get taxonomic annotations
tax_results = taxonomy.get_results(job_id)
```

---

## Motivation

Beak addresses common challenges in modeling evolutionary sequence relationships:

1. **Computational Resources**: Large-scale sequence searches and alignments are too intensive for personal laptops
2. **Workflow Integration**: Difficult to integrate remote computational steps into exploratory data analysis
3. **Reproducibility**: Parameters and workflows need to be traceable and comparable across projects

**Solution**: Beak works within your local notebook environment while seamlessly offloading compute-intensive tasks to remote servers. Search and alignment parameters are automatically cached, making analyses traceable and reproducible.

---

## Requirements

- Python 3.8+
- SSH access to a server with MMseqs2 installed
- Dependencies: `fabric`, `pandas`, `biopython`

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

[Your contact information]