import time
import sys
import shutil
from itertools import cycle

from pathlib import Path
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any, Callable
import pandas as pd

from .base import RemoteJobManager


class PipelineStep:
    """Represents a single step in a pipeline"""
    
    def __init__(self, step_type: str, step_name: str, params: Dict[str, Any], condition: Optional[Callable] = None):
        self.step_type = step_type  # 'search', 'align', 'taxonomy', etc.
        self.step_name = step_name  # User-friendly name for this step
        self.params = params
        self.condition = condition  # Optional condition function
        self.input_file = params.pop('input', None)  # Explicit input override
        self.output_name = params.pop('output', None)  # Explicit output name
    
    def should_run(self, context: Dict) -> bool:
        """Check if this step should run based on condition"""
        if self.condition is None:
            return True
        return self.condition(context)


class ConditionalBlock:
    """Handles conditional execution of pipeline steps"""
    
    def __init__(self, pipeline: 'Pipeline', condition: Callable):
        self.pipeline = pipeline
        self.condition = condition
        self.steps = []
    
    def then(self, step_type: str, **params):
        """Add a step to execute if condition is true"""
        step = PipelineStep(
            step_type=step_type,
            step_name=f"{step_type}_{len(self.pipeline.steps)}",
            params=params,
            condition=self.condition
        )
        self.pipeline.steps.append(step)
        return self.pipeline


class Pipeline(RemoteJobManager):
    """Pipeline orchestration for chaining bioinformatics jobs"""
    
    def __init__(self, host: str, user: str, key_path: Optional[str] = None, remote_job_dir: Optional[str] = None):
        super().__init__(host, user, key_path, remote_job_dir)
        self.steps: List[PipelineStep] = []
        self.initial_input = None

    def __repr__(self) -> str:
        """Return a nice description of the pipeline steps"""
        if not self.steps:
            return "Pipeline(no steps defined)"
        
        lines = ["Pipeline:"]
        lines.append(f"  Input: {self.initial_input or 'not set'}")
        lines.append(f"  Steps ({len(self.steps)}):")
        
        for i, step in enumerate(self.steps, 1):
            # Format step info
            step_desc = f"    {i}. {step.step_type}"
            
            # Add key parameters
            if step.params:
                key_params = []
                for k, v in step.params.items():
                    if k in ['database', 'model', 'threads', 'bootstrap', 'e']:
                        key_params.append(f"{k}={v}")
                if key_params:
                    step_desc += f" ({', '.join(key_params)})"
            
            # Add condition if present
            if step.condition:
                step_desc += " [conditional]"
            
            lines.append(step_desc)
        
        return "\n".join(lines)
    
    def debug_status(self, job_id: str):
        """Debug helper to see actual status file contents"""
        job_db = self._load_job_db()
        if job_id not in job_db:
            print(f"Job {job_id} not found in database")
            return
        
        remote_path = job_db[job_id]['remote_path']
        
        # Check if process is running
        pid = job_db[job_id]['pid']
        ps_result = self.conn.run(f'ps -p {pid}', warn=True, hide=False)
        
        status = self.conn.run(f'cat {remote_path}/status.txt 2>&1', warn=True, hide=False)
    
        files = self.conn.run(f'ls -la {remote_path}/', warn=True, hide=False)
        
        # Check pipeline script output
        # print("\nPipeline output (last 50 lines):")
        output = self.conn.run(f'tail -50 {remote_path}/nohup.out 2>&1', warn=True, hide=False)
        # print(output.stdout)
        
    def search(self, query_file: str = None, database: str = None, **params) -> 'Pipeline':
        """Add MMseqs2 search step"""
        if query_file:
            self.initial_input = query_file
        
        step = PipelineStep(
            step_type='search',
            step_name=f"search_{len(self.steps)}",
            params={'database': database, **params}
        )
        self.steps.append(step)
        return self
    
    def filter(self, 
           size: Optional[tuple] = None,
           motif: Optional[str] = None, 
           ref_seq_id: Optional[float] = None,
           taxonomic_level: Optional[str] = None,  # e.g., "Bacteria", "Eukaryota"
           min_quality: Optional[float] = None,  # based on alignment score/e-value
           max_sequences: Optional[int] = None,  # limit to top N
           deduplicate: bool = False,  # remove 100% identical sequences
           remove_fragments: bool = False,  # remove sequences with too many gaps/X
           **params) -> 'Pipeline':
        """
        Add sequence filtering step
        
        Args:
            size: (min_length, max_length) tuple in amino acids
            motif: Regex pattern that must be present (e.g., "C.{2,4}C" for zinc finger)
            ref_seq_id: Minimum % identity to reference sequence (requires reference)
            taxonomic_level: Keep only sequences from this taxonomic group (requires taxonomy step)
            min_quality: Minimum quality score (e.g., e-value threshold)
            max_sequences: Keep only top N sequences (by score/length)
            deduplicate: Remove 100% identical sequences
            remove_fragments: Remove sequences with >X% gaps or ambiguous residues
            **params: Additional filter parameters
        
        Returns:
            self for method chaining
        """
        step = PipelineStep(
            step_type='filter',
            step_name=f"filter_{len(self.steps)}",
            params={
                'size': size,
                'motif': motif,
                'ref_seq_id': ref_seq_id,
                'taxonomic_level': taxonomic_level,
                'min_quality': min_quality,
                'max_sequences': max_sequences,
                'deduplicate': deduplicate,
                'remove_fragments': remove_fragments,
                **params
            }
        )
        self.steps.append(step)
        return self
    
    def taxonomy(self, database: str = None, **params) -> 'Pipeline':
        """Add taxonomy annotation step"""
        step = PipelineStep(
            step_type='taxonomy',
            step_name=f"taxonomy_{len(self.steps)}",
            params={'database': database, **params}
        )
        self.steps.append(step)
        return self
    
    def align(self, **params) -> 'Pipeline':
        """Add Clustal Omega alignment step"""
        step = PipelineStep(
            step_type='align',
            step_name=f"align_{len(self.steps)}",
            params=params
        )
        self.steps.append(step)
        return self
    
    def tree(self, **params) -> 'Pipeline':
        """Add IQTree phylogenetic tree step"""
        step = PipelineStep(
            step_type='tree',
            step_name=f"tree_{len(self.steps)}",
            params=params
        )
        self.steps.append(step)
        return self
    
    def structure(self, **params) -> 'Pipeline':
        """Add structure prediction step"""
        step = PipelineStep(
            step_type='structure',
            step_name=f"structure_{len(self.steps)}",
            params=params
        )
        self.steps.append(step)
        return self
    
    def embeddings(self, model: str = 'esm2', **params) -> 'Pipeline':
        """Add sequence embedding step"""
        step = PipelineStep(
            step_type='embeddings',
            step_name=f"embeddings_{len(self.steps)}",
            params={'model': model, **params}
        )
        self.steps.append(step)
        return self
    
    # Conditional execution methods
    
    def if_min_hits(self, min_count: int) -> ConditionalBlock:
        """Execute following steps only if search has minimum hits"""
        def condition(context):
            hit_count = context.get('hit_count', 0)
            return hit_count >= min_count
        return ConditionalBlock(self, condition)
    
    def if_max_hits(self, max_count: int) -> ConditionalBlock:
        """Execute following steps only if search has fewer than max hits"""
        def condition(context):
            hit_count = context.get('hit_count', 0)
            return hit_count <= max_count
        return ConditionalBlock(self, condition)
    
    def if_condition(self, condition_func: Callable) -> ConditionalBlock:
        """Execute following steps based on custom condition"""
        return ConditionalBlock(self, condition_func)
    
    def _generate_pipeline_script(self, job_id: str, remote_path: str) -> str:
        """Generate bash script for entire pipeline"""
        
        script_parts = [
            "#!/bin/bash",
            "set -e",
            "",
            "# Pipeline execution script",
            f"echo \"Pipeline started: $(date)\" > {remote_path}/status.txt",
            f"echo 'RUNNING' >> {remote_path}/status.txt",
            "",
            "# Initialize context",
            "declare -A CONTEXT",
            ""
        ]
        
        previous_output = "input.fasta"
        search_output = None  # Track search results separately

        for i, step in enumerate(self.steps, 1):
            step_dir = f"{remote_path}/{i:02d}_{step.step_type}"
            
            script_parts.append(f"# Step {i}: {step.step_type}")
            script_parts.append(f"mkdir -p {step_dir}")
            
            if step.condition:
                script_parts.append(f'if [ "${{CONTEXT[run_next]}}" = "true" ]; then')
            
            if step.step_type == 'search':
                script_parts.extend(self._generate_search_commands(
                    step, step_dir, previous_output, f"{i:02d}_{step.step_type}_output", remote_path
                ))
                search_output = f"{step_dir}/{i:02d}_{step.step_type}_output.fasta"
                previous_output = search_output
            elif step.step_type == 'filter':
                input_for_filter = search_output or previous_output
                script_parts.extend(self._generate_filter_commands(
                    step, step_dir, input_for_filter, f"{i:02d}_{step.step_type}_output", remote_path
                ))
                previous_output = f"{step_dir}/filtered.fasta"
                if search_output:  # Update search_output if we filtered search results
                    search_output = f"{step_dir}/filtered.fasta"
            elif step.step_type == 'taxonomy':
                # Taxonomy uses search output but doesn't change the sequence flow
                input_for_tax = search_output or previous_output
                script_parts.extend(self._generate_taxonomy_commands(
                    step, step_dir, input_for_tax, f"{i:02d}_{step.step_type}_output", remote_path
                ))
                # Don't update previous_output - taxonomy is a side branch
            elif step.step_type == 'align':
                # Align uses search output (sequences), not taxonomy output
                input_for_align = search_output or previous_output
                script_parts.extend(self._generate_align_commands(
                    step, step_dir, input_for_align, f"{i:02d}_{step.step_type}_output", remote_path
                ))
                output_format = step.params.get('output_format', 'fasta')
                previous_output = f"{step_dir}/alignment.{output_format}"

            
            if step.condition:
                script_parts.append("fi")
            
            script_parts.append("")
        
        # Finalize
        script_parts.extend([
            "# Pipeline completed",
            f"echo \"Pipeline completed: $(date)\" >> {remote_path}/status.txt",
            f"echo 'COMPLETED' >> {remote_path}/status.txt"
        ])
        
        return "\n".join(script_parts)
    
    def _generate_search_commands(self, step, step_dir, input_file, output_name, remote_path) -> List[str]:
        """Generate MMseqs2 search commands"""
        database = step.params.get('database', 'uniref90')
        
        # Use the AVAILABLE_DBS mapping from MMseqsSearch
        from .search import MMseqsSearch
        if database in MMseqsSearch.AVAILABLE_DBS:
            db_file = MMseqsSearch.AVAILABLE_DBS[database]
            db_path = f"{self.DB_BASE_PATH}/{db_file}"
        elif database.startswith('/'):
            db_path = database
        else:
            db_path = f"{self.DB_BASE_PATH}/{database}"
        
        # Handle preset - expand to actual parameters
        params_to_use = step.params.copy()
        
        if 'preset' in params_to_use:
            preset_name = params_to_use.pop('preset')
            
            # Get preset parameters
            if preset_name in MMseqsSearch.PRESETS:
                preset_params = MMseqsSearch.PRESETS[preset_name]['params'].copy()
                # Merge with any user-provided params (user params override preset)
                for k, v in preset_params.items():
                    if k not in params_to_use:
                        params_to_use[k] = v
        
        # Format parameters
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
    
    def _generate_filter_commands(self, step, step_dir, input_file, output_name, remote_path) -> List[str]:
        """Generate sequence filtering commands using Python/BioPython"""
        input_fasta = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file
        
        # Build Python filter script
        script = ["from Bio import SeqIO", "import re", ""]
        script.append(f"records = list(SeqIO.parse('{input_fasta}', 'fasta'))")
        
        # Size filter
        if step.params.get('size'):
            min_len, max_len = step.params['size']
            script.append(f"records = [r for r in records if {min_len} <= len(r.seq) <= {max_len}]")
        
        # Motif filter
        if step.params.get('motif'):
            motif = step.params['motif']
            script.append(f"records = [r for r in records if re.search(r'{motif}', str(r.seq))]")
        
        # Remove fragments (>10% gaps/X)
        if step.params.get('remove_fragments'):
            script.append("records = [r for r in records if (str(r.seq).count('-') + str(r.seq).count('X')) / len(r.seq) <= 0.1]")
        
        # Deduplicate
        if step.params.get('deduplicate'):
            script.append("seen = set()")
            script.append("unique = [r for r in records if not (str(r.seq) in seen or seen.add(str(r.seq)))]")
            script.append("records = unique")
        
        # Max sequences (keep longest)
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
    
    def _generate_taxonomy_commands(self, step, step_dir, input_file, output_name, remote_path) -> List[str]:
        """Generate MMseqs2 taxonomy commands"""
        from .taxonomy import MMseqsTaxonomy
        
        database = step.params.get('database', 'uniprotkb')
        db_path = (f"{self.DB_BASE_PATH}/{MMseqsTaxonomy.AVAILABLE_TAX_DBS[database]}" 
                if database in MMseqsTaxonomy.AVAILABLE_TAX_DBS 
                else database if database.startswith('/') 
                else f"{self.DB_BASE_PATH}/{database}")
        
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
    
    def _generate_align_commands(self, step, step_dir, input_file, output_name, remote_path) -> List[str]:
        """Generate Clustal Omega alignment commands"""
        output_format = step.params.get('output_format', 'fasta')
        
        # Format params (Clustal uses --key=value)
        params = [f"--{k.replace('_', '-')}={v}" 
                for k, v in step.params.items() if k != 'output_format']
        
        if output_format != 'fasta':
            params.append(f'--outfmt={output_format}')
        
        input_fasta = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file
        
        return [
            "# Check if seqkit is available, otherwise use awk",
            f"if command -v seqkit &> /dev/null; then",
            f"  # Filter using seqkit (removes sequences with length 0)",
            f"  seqkit seq -m 1 {input_fasta} > {step_dir}/filtered_input.fasta",
            f"else",
            f"  # Filter using awk",
            f"  awk '/^>/ {{if (seq) print header \"\\n\" seq; header=$0; seq=\"\"; next}} {{seq=seq $0}} END {{if (seq) print header \"\\n\" seq}}' {input_fasta} | \\",
            f"    awk 'BEGIN {{RS=\">\"; ORS=\"\"}} NR>1 {{split($0,a,\"\\n\"); seq=\"\"; for(i=2;i<=length(a);i++) seq=seq a[i]; if (length(seq)>0) print \">\" $0}}' > {step_dir}/filtered_input.fasta",
            f"fi",
            "",
            "# Count sequences",
            f"echo \"Sequences for alignment: $(grep -c '^>' {step_dir}/filtered_input.fasta || echo 0)\"",
            "",
            "# Run Clustal Omega alignment",
            f"clustalo -i {step_dir}/filtered_input.fasta -o {step_dir}/alignment.{output_format} {' '.join(params)}"
        ]
    
    def _generate_tree_commands(self, step, step_dir, input_file, output_name, remote_path) -> List[str]:
        """Generate tree commands - placeholder"""
        return [
            f"# Tree step - to be implemented",
            f"echo 'Tree placeholder' > {step_dir}/{output_name}.tree"
        ]
    
    def _generate_structure_commands(self, step, step_dir, input_file, output_name, remote_path) -> List[str]:
        """Generate structure prediction commands - placeholder"""
        return [
            f"# Structure step - to be implemented",
            f"echo 'Structure placeholder' > {step_dir}/{output_name}.pdb"
        ]
    
    def _generate_embeddings_commands(self, step, step_dir, input_file, output_name, remote_path) -> List[str]:
        """Generate embedding commands using Docker service"""
        model = step.params.get('model', 'esm2')
        input_fasta = f"{remote_path}/{input_file}" if not input_file.startswith('/') else input_file
        
        # Ensure Docker service is running (done once at pipeline start)
        # Commands will use docker compose exec
        
        return [
            f"# Generate embeddings using Docker service",
            f"cd {self.remote_job_dir}/docker",
            f"docker compose exec -T embeddings python -m beak.remote.embeddings generate \\",
            f"  --input {input_fasta} \\",
            f"  --output {step_dir}/{output_name}.pt \\",
            f"  --model {model}"
        ]
    
    def execute(self, job_name: Optional[str] = None) -> str:
        """Execute the pipeline on the remote server"""
        
        if not self.steps:
            raise ValueError("Pipeline has no steps. Add steps before executing.")
        
        if not self.initial_input:
            raise ValueError("No input file specified. Use .search(query_file=...) to set input.")
        
        # Check if pipeline needs Docker
        needs_docker = any(step.step_type == 'embeddings' for step in self.steps)
        if needs_docker:
            print("Setting up Docker service for embeddings...")
            self._ensure_docker_service('embeddings')
        
        job_id = str(uuid.uuid4())[:8]
        
        # CREATE LOCAL PROJECT - ADD THIS BLOCK
        project_dir = self.create_project(
            job_id=job_id,
            job_type='pipeline',
            name=job_name,
            query_file=self.initial_input
        )
        
        # Create step subdirectories in project
        for i, step in enumerate(self.steps, 1):
            step_dir = project_dir / f"{i:02d}_{step.step_type}"
            step_dir.mkdir(exist_ok=True)
        
        print(f"📁 Created pipeline: {project_dir.name}")
        print(f"   Steps: {' → '.join(s.step_type for s in self.steps)}")
        
        # Set remote job directory details
        job_name = job_name or f"pipeline_{job_id}"
        remote_job_path = f"{self.remote_job_dir}/{job_id}"

        # Create remote job directory
        self.conn.run(f'mkdir -p {remote_job_path}', hide=True)
        print(f"Created remote directory: {remote_job_path}")
        
        # Upload initial input file
        remote_input = f"{remote_job_path}/input.fasta"
        print(f"Uploading input file...")
        self.conn.put(self.initial_input, remote_input)
        
        # Generate pipeline script
        pipeline_script = self._generate_pipeline_script(job_id, remote_job_path)
        
        # Upload and execute pipeline script
        script_path = f"{remote_job_path}/pipeline.sh"
        self.conn.put(
            local=self._write_temp_script(pipeline_script),
            remote=script_path
        )
        self.conn.run(f'chmod +x {script_path}', hide=True)
        
        # Submit job in background
        result = self.conn.run(
            f'nohup {script_path} > {remote_job_path}/nohup.out 2>&1 & echo $!',
            hide=True
        )
        pid = result.stdout.strip()
        
        # Save PID
        self.conn.run(f'echo {pid} > {remote_job_path}/pid.txt', hide=True)

        # After executing, check the actual script on server
        result = self.conn.run(f'cat {remote_job_path}/pipeline.sh', hide=True)
        
        # Update local job database
        job_db = self._load_job_db()
        job_db[job_id] = {
            'job_type': 'pipeline',
            'name': job_name,
            'steps': [{'type': s.step_type, 'params': s.params} for s in self.steps],
            'remote_path': remote_job_path,
            'submitted_at': datetime.now().isoformat(),
            'status': 'SUBMITTED',
            'pid': pid
        }
        self._save_job_db(job_db)
        
        print(f"✓ Pipeline submitted: {job_name} (ID: {job_id})")
        print(f"  Steps: {' → '.join(s.step_type for s in self.steps)}")
        print(f"  PID: {pid}")
        
        return job_id
    
    def get_step_results(self, job_id: str, step_number: int) -> Path:
        """Download results from a specific pipeline step"""
        job_db = self._load_job_db()
        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")
        
        remote_path = job_db[job_id]['remote_path']
        step_info = job_db[job_id]['steps'][step_number - 1]
        step_type = step_info['type']
        
        step_dir = f"{remote_path}/{step_number:02d}_{step_type}"
        
        # Determine output file based on step type
        output_files = {
            'search': 'results.m8',
            'taxonomy': 'taxonomy.tsv',
            'align': 'alignment.fasta',
            'tree': 'tree.nwk',
            'structure': 'structures/',
            'embeddings': 'embeddings.pt'
        }
        
        output_file = output_files.get(step_type, 'output')
        local_file = Path(f"{job_id}_step{step_number}_{step_type}_{output_file}")
        
        self.conn.get(f"{step_dir}/{output_file}", str(local_file))
        print(f"✓ Downloaded step {step_number} results to {local_file}")
        
        return local_file

    def detailed_status(self, job_id: str) -> Dict:
        """
        Get detailed status of pipeline including individual step progress
        
        Returns:
            Dict with overall status and per-step information
        """
        job_db = self._load_job_db()
        
        if job_id not in job_db:
            return {'status': 'UNKNOWN', 'error': 'Job ID not found'}
        
        job_info = job_db[job_id]
        remote_path = job_info['remote_path']
        
        # Get overall status
        overall_status = self.status(job_id)
        
        # Check which steps have completed
        step_statuses = []
        
        for i, step_info in enumerate(job_info['steps'], 1):
            step_type = step_info['type']
            step_dir = f"{remote_path}/{i:02d}_{step_type}"
            
            # Check if step directory exists
            dir_check = self.conn.run(
                f'[ -d {step_dir} ] && echo "EXISTS" || echo "NOT_FOUND"',
                hide=True, warn=True
            )
            
            if dir_check.stdout.strip() == "NOT_FOUND":
                step_status = 'PENDING'
                details = None
            else:
                # Check for output files to determine completion
                output_files = {
                    'search': 'results.m8',
                    'taxonomy': 'taxonomy.tsv',
                    'filter': 'filtered.fasta',
                    'align': 'alignment.fasta',
                    'tree': 'tree.nwk',
                    'structure': 'structures',
                    'embeddings': 'embeddings.pt'
                }
                
                output_file = output_files.get(step_type, 'output')
                
                # For search, check if it's still running or completed
                if step_type == 'search':
                    # Check if results.m8 exists and has content
                    result_check = self.conn.run(
                        f'[ -f {step_dir}/results.m8 ] && wc -l < {step_dir}/results.m8 || echo "0"',
                        hide=True, warn=True
                    )
                    hit_count = result_check.stdout.strip()
                    
                    if hit_count != "0":
                        step_status = 'COMPLETED'
                        details = {'hits': int(hit_count)}
                    else:
                        # Check if tmp directory exists (search is running)
                        tmp_check = self.conn.run(
                            f'[ -d {step_dir}/tmp ] && echo "RUNNING" || echo "COMPLETED"',
                            hide=True, warn=True
                        )
                        step_status = 'RUNNING' if tmp_check.stdout.strip() == "RUNNING" else 'PENDING'
                        details = None
                else:
                    # For other steps, check if output exists
                    file_check = self.conn.run(
                        f'[ -e {step_dir}/{output_file} ] && echo "EXISTS" || echo "NOT_FOUND"',
                        hide=True, warn=True
                    )
                    
                    if file_check.stdout.strip() == "EXISTS":
                        step_status = 'COMPLETED'
                        details = {'output': output_file}
                    else:
                        # Step started but not complete
                        step_status = 'RUNNING'
                        details = None
            
            step_statuses.append({
                'step': i,
                'type': step_type,
                'status': step_status,
                'params': step_info.get('params', {}),
                'details': details
            })
        
        return {
            'job_id': job_id,
            'name': job_info['name'],
            'overall_status': overall_status['status'],
            'runtime': overall_status['runtime'],
            'steps': step_statuses
        }

    def print_detailed_status(self, job_id: str, watch=False, animation_frame=0):
        """
        Print a nicely formatted detailed status report
        
        Args:
            job_id: Job ID to check
            watch: If True, refresh every 1 second. If int, refresh at that interval.
            animation_frame: Internal counter for animation (used during watch mode)
        """
        from IPython.display import clear_output
        
        # Animation frames for running steps
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        # Determine if this is a watch call
        is_watching = watch is not False
        
        # Clear output in watch mode (works in notebooks)
        if is_watching and animation_frame > 0:
            clear_output(wait=True)
        
        status = self.detailed_status(job_id)
        
        print(f"\n{'='*60}")
        print(f"Pipeline: {status['name']} ({status['job_id']})")
        print(f"Status: {status['overall_status']} | Runtime: {status['runtime']}")
        print(f"{'='*60}")
        
        for step in status['steps']:
            # Status icon
            if step['status'] == 'COMPLETED':
                icon = '✓'
            elif step['status'] == 'RUNNING':
                # Use animated spinner for running steps
                icon = spinner[animation_frame % len(spinner)] if is_watching else '⟳'
            else:
                icon = '○'
            
            # Format step line
            step_line = f"{icon} Step {step['step']}: {step['type']}"
            
            # Add key params
            if step['params']:
                key_params = []
                for k, v in step['params'].items():
                    if k in ['database', 'model', 'threads', 'bootstrap', 'e']:
                        key_params.append(f"{k}={v}")
                if key_params:
                    step_line += f" ({', '.join(key_params)})"
            
            # Add status
            step_line += f" [{step['status']}]"
            
            print(f"  {step_line}")
            
            # Add details if available
            if step['details']:
                for k, v in step['details'].items():
                    print(f"    └─ {k}: {v}")
        
        print(f"{'='*60}\n")
        
        if is_watching:
            print("(Press Ctrl+C to stop watching)")
        
        # Handle watch mode
        if watch is not False:
            # Determine interval
            if watch is True:
                interval = 1
            elif isinstance(watch, int):
                interval = watch
            else:
                raise ValueError("watch must be True, False, or an integer")
            
            try:
                # Check if job is still running
                if status['overall_status'] in ['RUNNING', 'PENDING']:
                    time.sleep(interval)
                    # Recursive call with incremented animation frame
                    self.print_detailed_status(job_id, watch=watch, animation_frame=animation_frame + 1)
                else:
                    print(f"\n✓ Job {status['overall_status'].lower()}. Stopped watching.")
            except KeyboardInterrupt:
                print("\n\nStopped watching.")

    def get_results(self, job_id: str, steps: Optional[List[int]] = None) -> Path:
        """
        Download pipeline results to local project directory
        
        Args:
            job_id: Job identifier
            steps: Optional list of step numbers to download (default: all)
        
        Returns:
            Path to project directory
        """
        # Check job status
        status_info = self.status(job_id)
        if status_info['status'] not in ['COMPLETED', 'RUNNING']:
            raise ValueError(f"Job not ready (status: {status_info['status']})")
        
        # Get project directory
        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            # Shouldn't happen, but create if missing
            project_dir = self.create_project(job_id, 'pipeline')
        
        job_db = self._load_job_db()
        job_info = job_db[job_id]
        remote_path = job_info['remote_path']
        
        print(f"Downloading results to: {project_dir.name}")
        
        # Determine which steps to download
        steps_to_download = steps or range(1, len(job_info['steps']) + 1)
        
        # Download each step's results
        for i in steps_to_download:
            if i < 1 or i > len(job_info['steps']):
                print(f"Warning: Step {i} doesn't exist, skipping")
                continue
            
            step_info = job_info['steps'][i - 1]
            step_type = step_info['type']
            local_step_dir = project_dir / f"{i:02d}_{step_type}"
            remote_step_dir = f"{remote_path}/{i:02d}_{step_type}"
            
            # Check if step has completed
            check = self.conn.run(
                f'[ -d {remote_step_dir} ] && echo "EXISTS" || echo "NOT_FOUND"',
                hide=True, warn=True
            )
            
            if check.stdout.strip() == "NOT_FOUND":
                print(f"  ⊗ Step {i} ({step_type}): not started yet")
                continue
            
            # Map step types to files to download
            files_to_download = {
                'search': ['results.m8', f"{i:02d}_{step_type}_output.fasta"],
                'filter': ['filtered.fasta', 'filter.py'],
                'taxonomy': ['taxonomy.tsv'],
                'align': ['alignment.fasta', 'alignment.phy'],
                'tree': ['tree.nwk', 'tree.treefile'],
                'embeddings': ['embeddings.pt']
            }
            
            files = files_to_download.get(step_type, [])
            downloaded_count = 0
            
            for filename in files:
                remote_file = f"{remote_step_dir}/{filename}"
                local_file = local_step_dir / filename
                
                # Skip if already downloaded
                if local_file.exists():
                    continue
                
                # Check if file exists on remote
                check = self.conn.run(
                    f'[ -f {remote_file} ] && echo "EXISTS" || echo "NOT_FOUND"',
                    hide=True, warn=True
                )
                
                if check.stdout.strip() == "EXISTS":
                    self.conn.get(remote_file, str(local_file))
                    downloaded_count += 1
            
            if downloaded_count > 0:
                print(f"  ✓ Step {i} ({step_type}): {downloaded_count} file(s)")
            else:
                print(f"  ○ Step {i} ({step_type}): up to date")
        
        # Also download pipeline logs
        log_file = project_dir / "pipeline.log"
        if not log_file.exists():
            self.conn.get(f"{remote_path}/nohup.out", str(log_file))
            print(f"  ✓ Pipeline log")
        
        print(f"\n✓ Project: {project_dir}")
        return project_dir
    
    def get_step_results(self, job_id: str, step_number: int) -> Path:
        """
        Download results from a specific pipeline step to project directory
        
        Args:
            job_id: Job identifier
            step_number: Step number (1-indexed)
        
        Returns:
            Path to step results file or directory
        """
        # Get project directory
        project_dir = self.get_project_dir(job_id)
        if not project_dir:
            raise ValueError(f"No project found for {job_id}")
        
        job_db = self._load_job_db()
        if job_id not in job_db:
            raise ValueError(f"Job {job_id} not found")
        
        job_info = job_db[job_id]
        remote_path = job_info['remote_path']
        step_info = job_info['steps'][step_number - 1]
        step_type = step_info['type']
        
        remote_step_dir = f"{remote_path}/{step_number:02d}_{step_type}"
        local_step_dir = project_dir / f"{step_number:02d}_{step_type}"
        local_step_dir.mkdir(exist_ok=True)
        
        # Determine output files
        output_files = {
            'search': ['results.m8', f"{step_number:02d}_{step_type}_output.fasta"],
            'taxonomy': ['taxonomy.tsv'],
            'filter': ['filtered.fasta'],
            'align': ['alignment.fasta'],
            'tree': ['tree.nwk'],
            'embeddings': ['embeddings.pt']
        }
        
        files = output_files.get(step_type, ['output'])
        
        downloaded = []
        for filename in files:
            remote_file = f"{remote_step_dir}/{filename}"
            local_file = local_step_dir / filename
            
            check = self.conn.run(
                f'[ -f {remote_file} ] && echo "EXISTS" || echo "NOT_FOUND"',
                hide=True, warn=True
            )
            
            if check.stdout.strip() == "EXISTS":
                self.conn.get(remote_file, str(local_file))
                downloaded.append(local_file)
        
        if downloaded:
            print(f"✓ Downloaded step {step_number} ({step_type}) to {local_step_dir.name}")
            return downloaded[0] if len(downloaded) == 1 else local_step_dir
        else:
            raise ValueError(f"No results found for step {step_number}")

    def wait(self, job_id: str, check_interval: int = 30, verbose: bool = True, download: bool = True):
        """
        Wait for pipeline to complete and optionally download results
        
        Args:
            job_id: Job identifier
            check_interval: Seconds between checks
            verbose: Print progress updates
            download: Automatically download results when complete
        
        Returns:
            Path to project directory if download=True, else status string
        """
        if verbose:
            print(f"Waiting for pipeline {job_id}...")
        
        while True:
            status_info = self.status(job_id)
            
            if verbose:
                print(f"  [{status_info['runtime']}] {status_info['status']}")
            
            if status_info['status'] in ['COMPLETED', 'FAILED', 'UNKNOWN']:
                break
            
            time.sleep(check_interval)
        
        if verbose:
            if status_info['status'] == 'COMPLETED':
                print(f"✓ Pipeline completed in {status_info['runtime']}")
            else:
                print(f"✗ Pipeline {status_info['status'].lower()}")
        
        if download and status_info['status'] == 'COMPLETED':
            return self.get_results(job_id)
        
        return status_info['status']
    
