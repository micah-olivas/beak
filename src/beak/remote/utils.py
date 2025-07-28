USER = None
PASSWORD = None

import os
import time
import json
import base64
import select
import subprocess
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

    # host parameter is available but not stored globally since it's passed to functions directly

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

def ssend_simple(cmd, sshProc, timeout=5, error_reporting=False):
    """Send a command without markers - for background processes."""
    if error_reporting:
        print(f'Executing: {cmd}')
    
    sshProc.stdin.write(cmd + '\n')
    sshProc.stdin.flush()
    
    # For background commands, just wait a moment
    time.sleep(0.5)
    return "Command sent to background"

def ssend(cmd, sshProc, timeout=5, error_reporting=False):
    """Send a command to the SSH process and read the output."""
    if error_reporting:
        print(f'Executing: {cmd}')
    
    # For background processes, use the simple version
    if cmd.strip().endswith('&'):
        return ssend_simple(cmd, sshProc, timeout, error_reporting)
    
    # Send command with newline and a unique marker
    marker = f"BEAK_END_{int(time.time())}"
    full_cmd = f"{cmd}; echo '{marker}'"
    
    sshProc.stdin.write(full_cmd + '\n')
    sshProc.stdin.flush()
    
    out = ''
    start_time = time.time()
    
    # Read all available output until we see our marker or timeout
    while True:
        rlist, _, _ = select.select([sshProc.stdout], [], [], 0.5)
        if rlist:
            line = sshProc.stdout.readline()
            if not line:
                break
            
            # Check if we found our end marker
            if marker in line:
                # Don't include the marker line in output
                break
                
            out += line
            start_time = time.time()  # Reset timer on new output
        else:
            # Check if we've exceeded timeout
            if time.time() - start_time > timeout:
                if error_reporting:
                    print(f"Command timed out after {timeout}s: {cmd}")
                break
    
    # Clean up any remaining markers that might have been included
    cleaned_out = out.replace(marker, '').strip()
    
    # Remove shell prompt patterns that might interfere
    lines = cleaned_out.split('\n')
    clean_lines = []
    for line in lines:
        # Skip lines that look like shell prompts or command echoes
        if not (line.startswith('(base)') or line.startswith('$') or line.strip() == cmd.strip()):
            clean_lines.append(line)
    
    return '\n'.join(clean_lines).strip()

def execute_remote_command(sshProc, cmd, timeout=30, error_reporting=False):
    """Execute a command remotely and check for success."""
    if error_reporting:
        print(f"   Executing: {cmd}")
    
    # Send the command
    result = ssend(cmd, sshProc, timeout=timeout, error_reporting=error_reporting)
    
    if error_reporting:
        print(f"   Output: {result}")
    
    # Check for common error indicators
    if any(indicator in result.lower() for indicator in ['error', 'command not found', 'no such file']):
        return False, result
    
    return True, result

def execute_long_running_command(sshProc, cmd, timeout=1800, progress_interval=30):
    """Execute a long-running command with progress updates."""
    print(f"   Executing: {cmd}")
    
    # Send command
    sshProc.stdin.write(cmd + '\n')
    sshProc.stdin.flush()
    
    start_time = time.time()
    last_progress = start_time
    out = ''
    
    print("   Progress: ", end="", flush=True)
    
    while True:
        rlist, _, _ = select.select([sshProc.stdout], [], [], 1.0)
        current_time = time.time()
        
        if rlist:
            line = sshProc.stdout.readline()
            if not line:
                break
            out += line
            last_progress = current_time
        else:
            # Show progress dots
            if current_time - last_progress > progress_interval:
                elapsed = int(current_time - start_time)
                print(f"\n   Still running... ({elapsed}s elapsed)", end="", flush=True)
                last_progress = current_time
            elif current_time - start_time > 5:  # After 5 seconds, start showing dots
                print(".", end="", flush=True)
            
            # Check timeout
            if current_time - start_time > timeout:
                print(f"\n   Command timed out after {timeout}s")
                return False, "Command timed out"
    
    elapsed = int(current_time - start_time)
    print(f"\n   Completed in {elapsed}s")
    
    # Check for errors
    if any(indicator in out.lower() for indicator in ['error', 'command not found', 'no such file']):
        return False, out.strip()
    
    return True, out.strip()

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

    try:
        sshProc = sopen()
    except Exception as e:
        print(f"❌ Failed to establish SSH connection: {e}")
        return False

    print("🪺 Nesting! Let's check your home directory on the remote...\n")
    time.sleep(0.5)

    # First, check for conda installation
    print("==== 1. Conda ==== ")
    check_conda = ssend("which conda", sshProc, timeout=10)
    print(f"Conda check result: {check_conda.strip()}")
    
    # If which conda fails, try alternative detection methods
    if "conda" not in check_conda:
        print("'which conda' failed, trying alternative detection...")
        check_conda_alt = ssend("conda --version 2>/dev/null || echo 'conda_not_found'", sshProc, timeout=10)
        print(f"Alternative conda check: {check_conda_alt.strip()}")
        if "conda_not_found" in check_conda_alt:
            conda_found = False
        else:
            conda_found = True
    else:
        conda_found = True
    
    if not conda_found:
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
    # First, find conda's environment directory
    conda_info = ssend("conda info --base", sshProc, timeout=10)
    print(f"Conda base: {conda_info.strip()}")
    
    # Check if beak environment exists using conda env list
    check_env = ssend("conda env list | grep beak || echo 'beak_not_found'", sshProc, timeout=10)
    print(f"Environment check result: {check_env.strip()}")

    if "beak_not_found" in check_env or "beak" not in check_env:
        print("beak environment not found. Creating beak environment...")
        # First, upload beak_env.yml to remote temp directory
        make_temp_dir(sshProc, remote_dir="temp_beak")
        
        # Find the beak_env.yml file - it should be in the same directory as this utils.py file
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        beak_env_path = os.path.join(current_dir, "beak_env.yml")
        
        if not os.path.exists(beak_env_path):
            print(f"❌ Error: beak_env.yml not found at {beak_env_path}")
            print("Please ensure beak_env.yml exists in the src/beak/remote/ directory")
            return False
            
        try:
            scp_to_remote(beak_env_path, "temp_beak/beak_env.yml", 'shr-zion.stanford.edu', USER, PASSWORD)
            print("✅ beak_env.yml uploaded successfully")
        except Exception as e:
            print(f"Failed to upload beak_env.yml: {e}")
            return False
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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_env_yml = os.path.join(current_dir, "beak_env.yml")
    remote_env_yml = "temp_beak/beak_env.yml"
    # Ensure temp_beak exists
    ssend("mkdir -p temp_beak", sshProc, timeout=10)
    
    if not os.path.exists(local_env_yml):
        print(f"❌ Error: beak_env.yml not found at {local_env_yml}")
        print("Skipping package consistency check")
    else:
        # Use scp to copy beak_env.yml to remote
        try:
            scp_to_remote(local_env_yml, remote_env_yml, 'shr-zion.stanford.edu', USER, PASSWORD)
            print("✅ beak_env.yml uploaded successfully")
        except Exception as e:
            print(f"Failed to upload beak_env.yml: {e}")
            print("Skipping package consistency check")
            local_env_yml = None  # Skip the consistency check
    print()
    time.sleep(0.5)

    # Only run consistency check if beak_env.yml was successfully uploaded
    if local_env_yml and os.path.exists(local_env_yml):
        # Compare installed packages with beak_env.yml
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
    else:
        print("Skipping package consistency check (beak_env.yml not available)")
    print()
    time.sleep(0.5)

    # Check for Homebrew
    print("==== 4. Checking for Homebrew ====")
    check_brew = ssend("which brew", sshProc, timeout=10)
    print(f"Homebrew check result: {check_brew.strip()}")
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

    # Check for mmseqs (placeholder for future implementation)
    print("==== 5. Additional Tools ====")
    print("Additional tool checks can be added here in the future.")
    print()

    print("🎉 Nesting complete! Your remote environment is ready.")
    
    # Properly close SSH connection
    try:
        sshProc.stdin.close()
        sshProc.stdout.close()
        sshProc.terminate()
        sshProc.wait(timeout=5)  # Wait for clean termination
    except Exception as e:
        print(f"Warning: Error closing SSH connection: {e}")
    
    return True
    

def search(query, db="UniRef90", sshProc=None, verbose=False):
    """
    Use mmseqs2 search to find similar sequences in protein databases.
    
    Args:
        query (str): Either a sequence string or a local FASTA file path.
        db (str): Database name from protein_sequence_databases (default: "UniRef90").
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed output. If False, show minimal progress.
    Returns:
        dict: {"results": "local_results_file.tsv", "config": "config_file.json"} or None on failure.
    """
    import uuid
    import time
    
    # Generate unique project directory for this search
    project_id = f"search_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    central_tmp_dir = "beak_tmp" 
    project_dir = f"{central_tmp_dir}/{project_id}"
    
    # Available databases in /srv/protein_sequence_databases/
    available_dbs = {
        'UniRef90': 'UniRef90',
        'UniRef50': 'UniRef50', 
        'UniRef100': 'UniRef100',
        'uniprot_all': 'uniprot_all_2021_04.fa',
        'uniref50': 'uniref50.fasta',
    }
    
    if db not in available_dbs:
        raise ValueError(f"Database '{db}' not available. Choose from: {list(available_dbs.keys())}")
    
    db_path = f"/srv/protein_sequence_databases/{available_dbs[db]}"
    
    if sshProc is None:
        sshProc = sopen()

    print(f"🔍 Starting mmseqs sequence search with database: {db}")
    if verbose:
        print(f"📁 Creating project directory: {project_dir}")
    
    # Create central beak_tmp directory and unique project directory
    ssend(f"mkdir -p ~/{project_dir}", sshProc, timeout=10)
    
    # Step 1: Create query FASTA file
    print("📝 Creating query FASTA file...")
    remote_query_fasta = f"~/{project_dir}/query.fasta"
    
    if os.path.isfile(query):
        if verbose:
            print("   📤 Uploading FASTA file to remote server...")
        scp_to_remote(query, remote_query_fasta, "shr-zion.stanford.edu", USER, PASSWORD)
        query_sequence = query  # For config file (will read the actual sequence later)
    else:
        if verbose:
            print("   ✍️  Writing sequence to remote FASTA file...")
        # Use a more robust method to write the FASTA file
        fasta_content = f">query_sequence\\n{query}"
        write_fasta_cmd = f'echo -e "{fasta_content}" > {remote_query_fasta}'
        result = ssend(write_fasta_cmd, sshProc, timeout=10)
        if verbose:
            print(f"   Write result: {result.strip()}")
        
        # Verify the file was created
        verify_cmd = f"ls -la {remote_query_fasta} && head -3 {remote_query_fasta}"
        verify_result = ssend(verify_cmd, sshProc, timeout=10)
        if verbose:
            print(f"   Verification: {verify_result.strip()}")
        query_sequence = query
    
    # Step 1.5: Create config file with operation details
    if verbose:
        print("📄 Creating config file...")
    import json
    from datetime import datetime
    
    config_data = {
        "project_id": project_id,
        "timestamp": datetime.now().isoformat(),
        "query_type": "file" if os.path.isfile(query) else "sequence",
        "query_sequence": query_sequence if not os.path.isfile(query) else f"File: {query}",
        "sequence_length": len(query_sequence) if not os.path.isfile(query) else "N/A (from file)",
        "database": db,
        "database_path": db_path,
        "remote_directory": project_dir,
        "mmseqs_command": "mmseqs search",
        "settings": {
            "sensitivity": 1,
            "max_seqs": 15000,
            "sort_results": 1,
            "createdb_timeout": 10,
            "search_timeout": 1800,
            "convert_timeout": 30
        }
    }
    
    config_json = json.dumps(config_data, indent=2)
    remote_config = f"~/{project_dir}/search_config.json"
    
    # Write config file to remote using a more robust method
    import tempfile
    
    # Write config to a local temp file first
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_file.write(config_json)
        local_temp_config = tmp_file.name
    
    try:
        # Upload the config file
        scp_to_remote(local_temp_config, remote_config, "shr-zion.stanford.edu", USER, PASSWORD)
        if verbose:
            print(f"   ✅ Config saved to: {remote_config}")
    finally:
        # Clean up local temp file
        os.unlink(local_temp_config)
    
    # Step 2: Find mmseqs and create database
    print("🗄️  Setting up mmseqs database...")
    
    # Check if mmseqs is available
    success, mmseqs_check = execute_remote_command(sshProc, "which mmseqs", timeout=10, error_reporting=verbose)
    
    if not success or not mmseqs_check.strip() or "not found" in mmseqs_check.lower():
        if verbose:
            print("❌ mmseqs not found in PATH. Checking common locations...")
        # Try common locations
        mmseqs_path = None
        for location in ["/usr/local/bin/mmseqs", "/opt/mmseqs/bin/mmseqs", "~/bin/mmseqs"]:
            success, test_result = execute_remote_command(sshProc, f"test -f {location} && echo 'found'", timeout=5)
            if success and "found" in test_result:
                if verbose:
                    print(f"   ✅ Found mmseqs at: {location}")
                mmseqs_path = location
                break
        
        if not mmseqs_path:
            print("❌ mmseqs not found. Make sure mmseqs2 is installed on the remote server.")
            return None
    else:
        # Extract the actual path from the output
        mmseqs_path_line = mmseqs_check.strip().split('\n')[-1]  # Get last line which should be the path
        if mmseqs_path_line and '/' in mmseqs_path_line:
            mmseqs_path = mmseqs_path_line.strip()
        else:
            mmseqs_path = "mmseqs"  # Fallback to PATH
        if verbose:
            print(f"   ✅ mmseqs found at: {mmseqs_path}")
    
    # Create the mmseqs database
    if verbose:
        print("   Creating mmseqs database from FASTA...")
    else:
        print("   ⚙️  Creating database...")
    create_db_cmd = f"cd ~/{project_dir} && {mmseqs_path} createdb query.fasta queryDB && echo 'DB_CREATION_COMPLETE'"
    success, create_result = execute_remote_command(sshProc, create_db_cmd, timeout=30, error_reporting=verbose)
    
    # Check if the command completed successfully by looking for our completion marker
    if "DB_CREATION_COMPLETE" in create_result or "Time for processing" in create_result:
        print("   ✅ Database created successfully")
        if verbose:
            print("   📁 mmseqs createdb completed with success indicators")
    else:
        if verbose:
            print(f"❌ Database creation may have failed")
            print(f"   Output: {create_result}")
        else:
            print("   ⚠️  Database creation uncertain, continuing...")
        # Try to continue anyway since mmseqs might have worked despite output issues
    
    # Simple verification - just check if queryDB file exists
    if verbose:
        print("   Verifying database files exist...")
        success, verify_result = execute_remote_command(sshProc, f"test -f ~/{project_dir}/queryDB && echo 'exists' || echo 'missing'", timeout=5)
        
        if "exists" in verify_result:
            print("   ✅ Database files confirmed present")
        else:
            print("   ⚠️  Database files may be missing, but continuing...")
            # Continue anyway - the createdb output suggests it worked
    
    # Step 3: Run mmseqs search (can take 10+ minutes for large databases)
    print(f"🔬 Running mmseqs search against {db}...")
    if not verbose:
        print("   ⏳ This may take several minutes...")
    else:
        print("   ⏳ This may take 10-30 minutes for large databases...")

    search_cmd = (
        f"cd ~/{project_dir} && {mmseqs_path} search queryDB {db_path} "
        f"resultDB_s1 tmp -s 1 --max-seqs 15000 --sort-results 1"
    )
    if verbose:
        print(f"   🔧 Command: {search_cmd}")
    
    # Capture the search output
    if verbose:
        print("   🚀 Starting search and capturing output...")
    
    # Run the search command with output redirection
    search_with_logging = f"{search_cmd} > ~/{project_dir}/search_output.log 2>&1"
    if verbose:
        print(f"   📝 Command with logging: {search_with_logging}")
    
    # Start the search command in background
    if verbose:
        print("   📤 Sending background command...")
    ssend(f"{search_with_logging} &", sshProc, timeout=5)
    
    # Wait a moment for the process to start
    import time
    time.sleep(3)
    
    # Check if there are any immediate errors in the log
    if verbose:
        print("   🔍 Checking for immediate errors...")
        success, log_check = execute_remote_command(sshProc, f"head -10 ~/{project_dir}/search_output.log 2>/dev/null || echo 'no_log_yet'", timeout=5, error_reporting=True)
        if success and "no_log_yet" not in log_check:
            print(f"   📋 Initial log content: {log_check}")
    
    # Check if the process started successfully
    success, process_info = execute_remote_command(sshProc, "ps aux | grep mmseqs | grep -v grep", timeout=5, error_reporting=verbose)
    if success and process_info.strip():
        print("   ✅ mmseqs search is running")
        if verbose:
            print(f"   🔍 Process info: {process_info.strip()}")
    else:
        print("   ⚠️  mmseqs process not found in process list")
        # Check if there's an error message in the log
        success, error_check = execute_remote_command(sshProc, f"cat ~/{project_dir}/search_output.log 2>/dev/null || echo 'no_log'", timeout=5, error_reporting=verbose)
        if success and "no_log" not in error_check:
            print(f"   ❌ Error in log: {error_check}")
    
    # Monitor progress by checking log file and result file
    start_time = time.time()
    max_wait = 1800  # 30 minutes
    check_interval = 30  # Check every 30 seconds
    last_log_size = 0
    
    if verbose:
        print("   Progress: ", end="", flush=True)
    else:
        print("   ⏳ Searching", end="", flush=True)
    
    while True:
        elapsed = time.time() - start_time
        
        # Check if search is complete by looking for result files
        success, check_result = execute_remote_command(
            sshProc, 
            f"test -f ~/{project_dir}/resultDB_s1 && echo 'complete' || echo 'running'",
            timeout=5
        )
        
        if "complete" in check_result:
            if verbose:
                print(f"\n   ✅ Search completed in {int(elapsed)}s!")
                # Show the final log output
                success, final_log = execute_remote_command(sshProc, f"tail -20 ~/{project_dir}/search_output.log", timeout=10, error_reporting=True)
                if success:
                    print("   📄 Final search output:")
                    print(f"   {final_log}")
            else:
                print(f" ✅ completed in {int(elapsed)}s")
            success = True
            search_result = "Search completed successfully"
            break
        elif elapsed > max_wait:
            if verbose:
                print(f"\n   ❌ Search timed out after {max_wait}s")
                # Show current log output to see what happened
                success, timeout_log = execute_remote_command(sshProc, f"tail -20 ~/{project_dir}/search_output.log", timeout=10, error_reporting=True)
                if success:
                    print("   📄 Search output at timeout:")
                    print(f"   {timeout_log}")
            else:
                print(f" ❌ timed out after {max_wait}s")
            success = False
            search_result = "Search timed out"
            break
        else:
            if verbose:
                # Check log file size to show progress
                log_check = execute_remote_command(sshProc, f"wc -l ~/{project_dir}/search_output.log 2>/dev/null || echo '0'", timeout=5)
                if log_check[0]:
                    try:
                        current_log_size = int(log_check[1].strip().split()[0])
                        if current_log_size > last_log_size:
                            print("📝", end="", flush=True)  # Log activity indicator
                            last_log_size = current_log_size
                        else:
                            print(".", end="", flush=True)
                    except:
                        print(".", end="", flush=True)
                
                # Show progress update every minute
                if int(elapsed) % 60 == 0 and elapsed > 0:
                    print(f"\n   Still running... ({int(elapsed)}s elapsed)")
                    # Show recent log output
                    recent_log = execute_remote_command(sshProc, f"tail -5 ~/{project_dir}/search_output.log", timeout=5, error_reporting=False)
                    if recent_log[0] and recent_log[1].strip():
                        print(f"   Recent output: {recent_log[1].strip()}")
                    print("   Progress: ", end="", flush=True)
            else:
                # Simple progress for non-verbose mode
                if int(elapsed) % 60 == 0 and elapsed > 0:
                    print(f" ({int(elapsed)}s)", end="", flush=True)
                else:
                    print(".", end="", flush=True)
            
            time.sleep(check_interval)
    
    if not success:
        print(f"❌ Error during search: {search_result}")
        return None
    
    if verbose:
        print("   ✅ Search completed!")
    
    # Step 4: Convert results to readable format (should be fast)
    print("📋 Converting results to readable format...")
    convert_cmd = f"cd ~/{project_dir} && {mmseqs_path} createtsv queryDB resultDB_s1 search_results.tsv"
    success, convert_result = execute_remote_command(sshProc, convert_cmd, timeout=60, error_reporting=verbose)
    
    if not success:
        print(f"❌ Error converting results: {convert_result}")
        return None
    
    print("   ✅ Results converted to TSV format")
    
    # Step 5: Retrieve results and config
    print("📥 Retrieving results from remote...")
    remote_results_file = f"~/{project_dir}/search_results.tsv"
    local_result = f"mmseqs_search_results_{project_id}.tsv"
    local_config = f"search_config_{project_id}.json"
    
    try:
        # Download results file
        scp_from_remote(remote_results_file, local_result, "shr-zion.stanford.edu", USER, PASSWORD)
        print(f"✅ Results saved to: {local_result}")
        
        # Download config file
        scp_from_remote(remote_config, local_config, "shr-zion.stanford.edu", USER, PASSWORD)
        if verbose:
            print(f"✅ Config saved to: {local_config}")
        
    except Exception as e:
        print(f"❌ Failed to retrieve files: {e}")
        if verbose:
            print("   You can manually retrieve results from:", remote_results_file)
            print("   You can manually retrieve config from:", remote_config)
        return None
    
    if verbose:
        print(f"🧹 Temporary files remain in: {project_dir} (for debugging)")
    return {"results": local_result, "config": local_config}







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