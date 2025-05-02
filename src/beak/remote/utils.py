USER = None
PASSWORD = None

import os
import time
import json
import base64
import select
import subprocess
import tempfile
from getpass import getpass
from cryptography.fernet import Fernet

AUTH_FILE = os.path.expanduser("~/.beak_auth.json")
KEY_FILE = os.path.expanduser("~/.beak_auth.key")

def _get_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return key

def authenticate(user, host=None):
    global USER, PASSWORD
    USER = user
    key = _get_key()
    fernet = Fernet(key)

    if host is not None:
        HOST = host
    else:
        HOST = 'shr-zion.stanford.edu'

    # Try to load password from file
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        if user in data:
            enc_pwd = base64.b64decode(data[user])
            PASSWORD = fernet.decrypt(enc_pwd).decode()
            return

    # Prompt for password and store encrypted
    PASSWORD = getpass(f"Password for {user}: ")
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f:
            try:
                print("exists")
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    # Encrypt and encode the password before saving
    enc_pwd = fernet.encrypt(PASSWORD.encode())
    enc_pwd_b64 = base64.b64encode(enc_pwd).decode()
    data[user] = enc_pwd_b64
    with open(AUTH_FILE, "w") as f:
        json.dump(data, f)

def get_pw():
    """Retrieve and decrypt the password for the current USER from AUTH_FILE."""
    if USER is None:
        raise ValueError("USER is not set.")
    key = _get_key()
    fernet = Fernet(key)
    if os.path.exists(AUTH_FILE):
        with open(AUTH_FILE, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
        if USER in data:
            enc_pwd = base64.b64decode(data[USER])
            return fernet.decrypt(enc_pwd).decode()
    return None

def sopen():
    global PASSWORD, USER
    HOST = "shr-zion.stanford.edu"
    print(f"Connecting to {HOST}...")

    if USER is None or PASSWORD is None:
        user = input("Username: ") if USER is None else USER
        authenticate(user)

    sshProc = subprocess.Popen(
        ['sshpass', '-p', PASSWORD, 'ssh', '-tt', f'{USER}@{HOST}'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Wait briefly for any immediate connection errors
    time.sleep(1)
    if sshProc.poll() is not None:
        stderr_output = sshProc.stderr.read()
        raise RuntimeError(f"SSH connection failed: {stderr_output.strip()}")

    # Optionally, check for a prompt or banner to confirm connection
    ready, _, _ = select.select([sshProc.stdout], [], [], 2)
    if not ready:
        sshProc.terminate()
        raise RuntimeError("SSH connection timed out or did not respond.")
    banner = sshProc.stdout.readline()
    if not banner:
        sshProc.terminate()
        raise RuntimeError("SSH connection closed unexpectedly.")
    print("Connection successful\n")
    return sshProc

def ssend(cmd, sshProc, timeout=5, error_reporting=False):
    sshProc.stdin.write(cmd + '\n')
    if error_reporting:
        print(f'done with write! cmd: {cmd}')
    
    sshProc.stdin.flush()
    if error_reporting:
        print(f'done with flush! cmd: {cmd}')
    
    out = ''
    
    # Read all available output
    start_time = time.time()
    while True:
        rlist, _, _ = select.select([sshProc.stdout], [], [], 0.1)
        if rlist:
            line = sshProc.stdout.readline()
            if not line:
                break
            out += line
            start_time = time.time()  # Reset timer on new output
        else:
            if time.time() - start_time > timeout:
                break
    return out

def make_temp_dir(sshProc, remote_dir="temp_beak"):
    """Ensure the remote temp directory exists.
    """
    # Check if the directory exists first
    check_cmd = f'if [ -d "{remote_dir}" ]; then echo "exists"; else echo "not_exists"; fi'
    result = ssend(check_cmd, sshProc)
    if "not_exists" in result:
        ssend(f"mkdir -p {remote_dir}", sshProc)

def scp_to_remote(local_path, remote_path, HOST, USER, PASSWORD):
    """Copy a file from local_path to remote_path using scp and sshpass.
    """
    if USER is None or PASSWORD is None:
        user = input("Username: ") if USER is None else USER
        authenticate(user)

    scp_cmd = [
        "sshpass", "-p", PASSWORD,
        "scp", "-o", "StrictHostKeyChecking=no",
        local_path,
        f"{USER}@{HOST}:{remote_path}"
    ]
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SCP failed: {result.stderr}")
    return result

def scp_from_remote(remote_path, local_path, HOST, USER, PASSWORD):
    """Copy a file from remote_path to local_path using scp and sshpass.
    """
    if USER is None or PASSWORD is None:
        user = input("Username: ") if USER is None else USER
        authenticate(user)

    scp_cmd = [
        "sshpass", "-p", PASSWORD,
        "scp", "-o", "StrictHostKeyChecking=no",
        f"{USER}@{HOST}:{remote_path}",
        local_path
    ]
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SCP from remote failed: {result.stderr}")
    return result

def nest():
    """ Set up home environment
    """
    global HOST, PASSWORD

    if USER is None or PASSWORD is None:
        user = input("Username: ") if USER is None else USER
        authenticate(user)

    sshProc = sopen()

    print("🪺 Nesting! Let's check your home directory on the remote...\n")
    time.sleep(0.5)

    # First, check for conda installation
    print("==== 1. Conda ==== ")
    check_conda = ssend("which conda", sshProc)
    if "conda" not in check_conda:
        print("Conda is not installed on the remote server.")
        print("Installing Miniconda on the remote server...")
        install_cmd = (
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && "
            "bash ~/miniconda.sh -b -p $HOME/miniconda && "
            "rm ~/miniconda.sh && "
            "echo 'export PATH=\"$HOME/miniconda/bin:$PATH\"' >> ~/.bashrc && "
            "source ~/.bashrc"
        )
        ssend(install_cmd, sshProc, timeout=10)
        print("Miniconda installation complete.")
    else:
        print("Conda is already installed on the remote server.")
    print()
    time.sleep(0.5)

    # Now check for beak conda environment
    print("==== 2. Check beak Conda Environment ==== ")
    # Use a shorter timeout to avoid stalling if the remote command hangs
    print('got here')
    check_env = ssend("ls $HOME/anaconda3/envs", sshProc, error_reporting=True)

    if "beak" not in check_env:
        print("beak environment not found. Creating beak environment...")
        # First, upload beak_env.yml to remote temp directory
        make_temp_dir(sshProc, remote_dir="temp_beak")
        try:
            scp_to_remote("beak_env.yml", "temp_beak/beak_env.yml", 'shr-zion.stanford.edu', USER, PASSWORD)
        except Exception as e:
            print(f"Failed to upload beak_env.yml: {e}")
            return
        create_env_cmd = f"conda env create -n beak -f temp_beak/beak_env.yml"
        ssend(create_env_cmd, sshProc, timeout=30)
        print("beak environment created!")
    else:
        print("beak environment already exists!")
    print()
    time.sleep(0.5)

    # 3. Check packages in beak environment against beak_env.yml
    print("==== 3. Package Consistency Check ==== ")
    # Upload beak_env.yml to remote temp directory
    local_env_yml = "beak_env.yml"
    remote_env_yml = "temp_beak/beak_env.yml"
    # Ensure temp_beak exists
    print('got here')
    ssend("mkdir -p temp_beak", sshProc, error_reporting=True)
    # Use scp to copy beak_env.yml to remote
    try:
        scp_to_remote(local_env_yml, remote_env_yml, 'shr-zion.stanford.edu', USER, PASSWORD)
    except Exception as e:
        print(f"Failed to upload beak_env.yml: {e}")
        return
    print()
    time.sleep(0.5)

    # List installed packages in beak environment
    list_pkgs_cmd = "source activate beak && conda list --export"
    installed_pkgs = ssend(list_pkgs_cmd, sshProc, timeout=10)
    # Compare with beak_env.yml
    compare_cmd = (
        "conda env export -n beak > temp_beak/current_env.yml && "
        "diff temp_beak/current_env.yml temp_beak/beak_env.yml || true"
    )
    diff_output = ssend(compare_cmd, sshProc, timeout=10)
    if diff_output.strip():
        print("Differences found between beak environment and beak_env.yml:")
        print(diff_output)
    else:
        print("beak environment matches beak_env.yml!")
    print()
    time.sleep(0.5)

    # Check for Homebrew
    print("==== 4. Checking for Homebrew ====")
    check_brew = ssend("which brew", sshProc)
    if "brew" not in check_brew:
        print("Homebrew is not installed on the remote server.")
        print("Installing Homebrew on the remote server...")
        ssend('/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"', sshProc, timeout=60)
        ssend('echo "eval \"\$($(brew --prefix)/bin/brew shellenv)\"" >> ~/.bashrc', sshProc, timeout=60)
        print("Homebrew installation complete.")
    else:
        print("Homebrew is already installed on the remote server.")
    print()
    time.sleep(1)

    # Check for mmseqs


    sshProc.stdin.close()
    sshProc.stdout.close()
    sshProc.terminate()
    

def search(query, db=None, remote_dir="temp_beak", sshProc=None):
    """
    Use mmseqs2 to search for similar sequences.
    If a single sequence is provided, generate a FASTA file on the remote.
    If a FASTA file path is provided, use that.
    Args:
        query (str): Either a sequence string or a local FASTA file path.
        db_path (str): Path to the mmseqs2 database on the remote server.
        remote_dir (str): Directory on remote to use for temp files.
        sshProc: An open SSH process (from sopen()).
    Returns:
        str: mmseqs2 search output (stdout).
    """

    db_dict = {
        'uniprot_all': 'uniprot_all_2021_04.fa',
        'uniref50': 'uniref50.fasta',
        
    }
    
    db_path = "/srv/protein_sequence_databases" + db_dict[db]

    if sshProc is None:
        sshProc = sopen()

    print("Ensuring remote temp directory exists...")
    make_temp_dir(sshProc, remote_dir=remote_dir)

    # Determine if query is a sequence or a file
    if os.path.isfile(query):
        print("Uploading FASTA file to remote server...")
        remote_query = f"{remote_dir}/query.fasta"
        scp_to_remote(query, remote_query, "shr-zion.stanford.edu", USER, PASSWORD)
    else:
        print("Writing sequence to remote FASTA file...")
        remote_query = f"{remote_dir}/query.fasta"
        heredoc = f"cat << 'EOF' > {remote_query}\n>query\n{query}\nEOF"
        ssend(heredoc, sshProc)

    print("Preparing output directory on remote...")
    remote_out = f"{remote_dir}/mmseqs_out"
    ssend(f"rm -rf {remote_out}", sshProc)
    ssend(f"mkdir -p {remote_out}", sshProc)

    print("Running mmseqs2 search on remote...")
    search_cmd = (
        f"mmseqs easy-search {remote_query} {db_path} {remote_out}/result.m8 "
        f"--format-output 'query,target,pident,evalue,qcov,tcov' -v 1"
    )
    output = ssend(search_cmd, sshProc, timeout=60)

    print("Retrieving result file from remote...")
    local_result = "mmseqs_result.m8"
    scp_from_remote(f"{remote_out}/result.m8", local_result, "shr-zion.stanford.edu", USER, PASSWORD)

    print(f"Results saved to {local_result}")
    return output







# def esm_single_sequence(
#     sequence: str,
#     script_filename: str = "run_esm.py",
# ) -> RemoteResult:
#     """Upload & run an ESM inference script on `sequence`, returning a RemoteResult."""
#     ESM_SCRIPT = f"""\
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig

# protein = ESMProtein(sequence="{sequence}")
# client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
# protein_tensor = client.encode(protein)
# logits_output = client.logits(
#    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
# )
# logits_output
# """
#     global HOST
#     ensure_remote_dir(remote_dir, HOST, USER, PASSWORD) # make temp
#     heredoc = (
#         f"cat << 'EOF' > temp_beak/{script_filename}\n"
#         f"{ESM_SCRIPT}\n"
#         "EOF"
#     )
#     prep = send(heredoc)
#     if prep.returncode != 0:
#         raise RuntimeError(f"Failed to upload script:\n{prep.stderr}")

#     return send(f"python3 temp_beak/{script_filename}")




# def ensure_remote_dir(remote_dir: str, HOST: str, USER: str, PASSWORD: str):
#     res = run_remote(f"if [ ! -d {remote_dir} ]; then mkdir -p {remote_dir}; fi", HOST, USER, PASSWORD)
#     if res.returncode != 0:
#         raise RuntimeError(f"Could not create {remote_dir!r}:\n{res.stderr}")


# def esm_single_sequence(
#     sequence: str,
#     script_filename: str = "run_esm.py",
# ) -> RemoteResult:
#     """Upload & run an ESM inference script on `sequence`, returning a RemoteResult."""
#     ESM_SCRIPT = f"""\
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
# import base64
# import json
# from cryptography.fernet import Fernet

# protein = ESMProtein(sequence="{sequence}")
# client = ESMC.from_pretrained("esmc_300m").to("cuda")  # or "cpu"
# protein_tensor = client.encode(protein)
# logits_output = client.logits(
#    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
# )
# logits_output
# """
#     # # 1) make the temp directory
#     # ensure_remote_dir(remote_dir, HOST, USER, PASSWORD)

#     # 2) upload the script
#     heredoc = (
#         f"cat << 'EOF' > temp_beak/{script_filename}\n"
#         f"{ESM_SCRIPT}\n"
#         "EOF"
#     )
#     prep = run_remote(heredoc)
#     if prep.returncode != 0:
#         raise RuntimeError(f"Failed to upload script:\n{prep.stderr}")

#     # 3) run it
#     return run_remote(f"python temp_beak/{script_filename}")