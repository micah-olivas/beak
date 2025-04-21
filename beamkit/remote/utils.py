import os
import subprocess
import tempfile

class RemoteResult:
    """Wraps remote stdout/stderr so notebooks render them nicely."""
    def __init__(self, stdout: str, stderr: str, returncode: int):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode

    def __repr__(self):
        out = f"=== STDOUT ===\n{self.stdout}"
        if self.stderr:
            out += f"\n=== STDERR ===\n{self.stderr}"
        return out

def run_remote(cmd: str, HOST: str, USER: str, PASSWORD: str) -> RemoteResult:
    conda_setup = "/home/mbolivas/anaconda3/etc/profile.d/conda.sh"
    full_cmd = f"source {conda_setup} && conda activate ESM && {cmd}"
    args = ["sshpass", "-p", PASSWORD, "ssh", f"{USER}@{HOST}", full_cmd]
    res = subprocess.run(args, capture_output=True, text=True)
    return RemoteResult(res.stdout, res.stderr, res.returncode)

def ensure_remote_dir(remote_dir: str, HOST: str, USER: str, PASSWORD: str):
    res = run_remote(f"mkdir -p {remote_dir}", HOST, USER, PASSWORD)
    if res.returncode != 0:
        raise RuntimeError(f"Could not create {remote_dir!r}:\n{res.stderr}")

def scp_from_remote(remote_path: str, local_path: str, HOST: str, USER: str, PASSWORD: str):
    """Copy a file from remote_path on HOST to local_path."""
    args = [
        "sshpass", "-p", PASSWORD,
        "scp", f"{USER}@{HOST}:{remote_path}", local_path
    ]
    subprocess.run(args, check=True)

def deploy_and_run_esm(
    sequence: str,
    HOST: str,
    USER: str,
    PASSWORD: str,
    remote_dir: str = "temp_bk",
    script_filename: str = "run_esm.py",
) -> RemoteResult:
    """Upload & run an ESM inference script on `sequence`, returning a RemoteResult."""
    ESM_SCRIPT = f"""\
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

protein = ESMProtein(sequence="{sequence}")
client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
protein_tensor = client.encode(protein)
logits_output = client.logits(
   protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
)
logits_output
"""
    # 1) make the temp directory
    ensure_remote_dir(remote_dir, HOST, USER, PASSWORD)

    # 2) upload the script
    heredoc = (
        f"cat << 'EOF' > {remote_dir}/{script_filename}\n"
        f"{ESM_SCRIPT}\n"
        "EOF"
    )
    prep = run_remote(heredoc, HOST, USER, PASSWORD)
    if prep.returncode != 0:
        raise RuntimeError(f"Failed to upload script:\n{prep.stderr}")

    # 3) run it
    return run_remote(f"python {remote_dir}/{script_filename}", HOST, USER, PASSWORD)

def connect_remote