"""
Interactive remote utilities optimized for Jupyter notebooks
"""

import time
import threading
import queue
from typing import Optional, Callable, Generator
from IPython.display import display, clear_output
import ipywidgets as widgets

from .utils import sopen, ssend, make_temp_dir, scp_to_remote, scp_from_remote, USER, PASSWORD


class InteractiveRemoteSession:
    """Non-blocking remote session for Jupyter notebooks"""
    
    def __init__(self, host: str = "shr-zion.stanford.edu"):
        self.host = host
        self.ssh_proc = None
        self.is_connected = False
        self._command_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._worker_thread = None
        
    def connect(self, user: str = None) -> None:
        """Connect to remote server with progress indicator"""
        if user:
            from .utils import authenticate
            authenticate(user, self.host)
            
        progress = widgets.IntProgress(
            value=0, min=0, max=3,
            description='Connecting:',
            bar_style='info'
        )
        status = widgets.HTML(value="Initializing connection...")
        display(widgets.VBox([progress, status]))
        
        try:
            progress.value = 1
            status.value = "Establishing SSH connection..."
            self.ssh_proc = sopen()
            
            progress.value = 2
            status.value = "Verifying connection..."
            test_result = ssend("echo 'connected'", self.ssh_proc, timeout=5)
            
            if "connected" in test_result:
                progress.value = 3
                progress.bar_style = 'success'
                status.value = "✅ Connected successfully!"
                self.is_connected = True
                self._start_worker_thread()
            else:
                raise RuntimeError("Connection test failed")
                
        except Exception as e:
            progress.bar_style = 'danger'
            status.value = f"❌ Connection failed: {str(e)}"
            raise
    
    def _start_worker_thread(self):
        """Start background thread for command execution"""
        self._worker_thread = threading.Thread(target=self._command_worker, daemon=True)
        self._worker_thread.start()
    
    def _command_worker(self):
        """Background worker for executing commands"""
        while self.is_connected:
            try:
                command, timeout, callback = self._command_queue.get(timeout=1)
                result = ssend(command, self.ssh_proc, timeout=timeout)
                self._result_queue.put((callback, result, None))
            except queue.Empty:
                continue
            except Exception as e:
                self._result_queue.put((None, None, e))
    
    def run_command_async(self, command: str, timeout: int = 30, 
                         callback: Optional[Callable] = None) -> None:
        """Execute command asynchronously"""
        if not self.is_connected:
            raise RuntimeError("Not connected to remote server")
        
        self._command_queue.put((command, timeout, callback))
    
    def run_command_with_progress(self, command: str, description: str = "Running command...",
                                timeout: int = 30) -> str:
        """Execute command with live progress indicator"""
        if not self.is_connected:
            raise RuntimeError("Not connected to remote server")
        
        progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description=description,
            bar_style='info'
        )
        status = widgets.HTML(value="Starting...")
        output_area = widgets.Output()
        
        display(widgets.VBox([progress, status, output_area]))
        
        # Simulate progress while command runs
        def update_progress():
            for i in range(1, 101):
                time.sleep(timeout / 100)
                progress.value = i
                if i < 50:
                    status.value = "Executing command..."
                elif i < 90:
                    status.value = "Processing output..."
                else:
                    status.value = "Finishing up..."
        
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
        
        try:
            result = ssend(command, self.ssh_proc, timeout=timeout)
            progress.value = 100
            progress.bar_style = 'success'
            status.value = "✅ Command completed"
            
            with output_area:
                print("Command output:")
                print(result)
            
            return result
            
        except Exception as e:
            progress.bar_style = 'danger'
            status.value = f"❌ Command failed: {str(e)}"
            raise
    
    def nest_interactive(self) -> None:
        """Interactive version of nest() with progress tracking"""
        if not self.is_connected:
            raise RuntimeError("Not connected to remote server")
        
        steps = [
            ("Checking conda installation", "which conda", 10),
            ("Checking beak environment", "ls $HOME/anaconda3/envs", 10),
            ("Verifying package consistency", "conda env export -n beak", 15),
            ("Checking homebrew", "which brew", 5)
        ]
        
        overall_progress = widgets.IntProgress(
            value=0, min=0, max=len(steps),
            description='Setup Progress:',
            bar_style='info'
        )
        current_step = widgets.HTML(value="Starting setup...")
        output_area = widgets.Output()
        
        display(widgets.VBox([overall_progress, current_step, output_area]))
        
        for i, (desc, cmd, timeout) in enumerate(steps):
            current_step.value = f"Step {i+1}/{len(steps)}: {desc}"
            
            with output_area:
                print(f"\n🔄 {desc}")
                
            try:
                result = ssend(cmd, self.ssh_proc, timeout=timeout)
                with output_area:
                    print(f"✅ {desc} - Complete")
                    if result.strip():
                        print(f"Output: {result[:200]}...")
                        
            except Exception as e:
                with output_area:
                    print(f"❌ {desc} - Error: {str(e)}")
                    
            overall_progress.value = i + 1
        
        overall_progress.bar_style = 'success'
        current_step.value = "✅ Setup complete!"
    
    def search_interactive(self, query: str, db: str = "uniprot_all",
                          remote_dir: str = "temp_beak") -> str:
        """Interactive sequence search with progress tracking"""
        if not self.is_connected:
            raise RuntimeError("Not connected to remote server")
        
        steps = [
            "Preparing remote directory",
            "Uploading query sequence",
            "Running mmseqs2 search", 
            "Retrieving results"
        ]
        
        progress = widgets.IntProgress(
            value=0, min=0, max=len(steps),
            description='Search Progress:',
            bar_style='info'
        )
        status = widgets.HTML(value="Starting search...")
        output_area = widgets.Output()
        
        display(widgets.VBox([progress, status, output_area]))
        
        try:
            # Step 1: Prepare directory
            status.value = f"Step 1/{len(steps)}: {steps[0]}"
            make_temp_dir(self.ssh_proc, remote_dir=remote_dir)
            progress.value = 1
            
            with output_area:
                print("✅ Remote directory prepared")
            
            # Step 2: Upload query
            status.value = f"Step 2/{len(steps)}: {steps[1]}"
            remote_query = f"{remote_dir}/query.fasta"
            heredoc = f"cat << 'EOF' > {remote_query}\n>query\n{query}\nEOF"
            ssend(heredoc, self.ssh_proc)
            progress.value = 2
            
            with output_area:
                print("✅ Query sequence uploaded")
            
            # Step 3: Run search
            status.value = f"Step 3/{len(steps)}: {steps[2]}"
            db_dict = {
                'uniprot_all': 'uniprot_all_2021_04.fa',
                'uniref50': 'uniref50.fasta',
            }
            db_path = f"/srv/protein_sequence_databases/{db_dict[db]}"
            remote_out = f"{remote_dir}/mmseqs_out"
            
            ssend(f"rm -rf {remote_out}", self.ssh_proc)
            ssend(f"mkdir -p {remote_out}", self.ssh_proc)
            
            search_cmd = (
                f"mmseqs easy-search {remote_query} {db_path} {remote_out}/result.m8 "
                f"--format-output 'query,target,pident,evalue,qcov,tcov' -v 1"
            )
            
            with output_area:
                print("🔄 Running mmseqs2 search (this may take a while)...")
            
            result = ssend(search_cmd, self.ssh_proc, timeout=120)
            progress.value = 3
            
            with output_area:
                print("✅ Search completed")
            
            # Step 4: Retrieve results
            status.value = f"Step 4/{len(steps)}: {steps[3]}"
            local_result = "mmseqs_result.m8"
            scp_from_remote(f"{remote_out}/result.m8", local_result, 
                          self.host, USER, PASSWORD)
            
            progress.value = 4
            progress.bar_style = 'success'
            status.value = "✅ Search complete!"
            
            with output_area:
                print(f"✅ Results saved to {local_result}")
            
            return result
            
        except Exception as e:
            progress.bar_style = 'danger'
            status.value = f"❌ Search failed: {str(e)}"
            with output_area:
                print(f"Error: {str(e)}")
            raise
    
    def disconnect(self):
        """Clean up connection"""
        self.is_connected = False
        if self.ssh_proc:
            self.ssh_proc.terminate()
            self.ssh_proc = None


# Convenience functions for notebook use
def connect_remote(user: str = None, host: str = "shr-zion.stanford.edu") -> InteractiveRemoteSession:
    """Quick connect function for notebooks"""
    session = InteractiveRemoteSession(host)
    session.connect(user)
    return session