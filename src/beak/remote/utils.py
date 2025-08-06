USER = None
PASSWORD = None

# Database cache to avoid repeated server checks
_DATABASE_CACHE = {}
_DATABASE_CACHE_TIMESTAMP = None
_DATABASE_CACHE_TIMEOUT = 300  # 5 minutes

import os
import time
import json
import base64
import select
import subprocess
import shutil
from getpass import getpass
from cryptography.fernet import Fernet
from gibberish import Gibberish

AUTH_FILE = os.path.expanduser("~/.beak_auth.json")
KEY_FILE = os.path.expanduser("~/.beak_auth.key")

def _generate_project_id(project_id=None):
    """
    Generate a memorable project ID using either user-provided ID or 2 gibberish words.
    
    Args:
        project_id (str, optional): User-provided project identifier
    
    Returns:
        str: Project ID in format 'beak_user_id' or 'beak_word1_word2'
    """
    if project_id:
        # Clean project ID to be filesystem-safe
        clean_id = ''.join(c for c in project_id if c.isalnum() or c in '-_').lower()
        return f"beak_{clean_id}"
    
    # Generate 2 gibberish words
    try:
        gib = Gibberish()
        words = gib.generate_words(2)
        
        if len(words) == 2:
            return f"beak_{'_'.join(words)}"
        else:
            # Fallback to timestamp if gibberish fails
            return f"beak_{int(time.time())}"
    except Exception:
        # Fallback to timestamp if gibberish fails
        return f"beak_{int(time.time())}"


def _get_manifest_path():
    """Get the path to the job manifest file."""
    return "beak_tmp/.beak_jobs.json"


def _load_job_manifest(sshProc):
    """Load the job manifest from remote server."""
    manifest_path = _get_manifest_path()
    
    # Check if manifest exists
    success, exists_check = execute_remote_command(sshProc, f"test -f ~/{manifest_path} && echo 'exists' || echo 'not_exists'", timeout=5)
    
    if "not_exists" in exists_check:
        # Create empty manifest
        empty_manifest = {
            "jobs": {},
            "last_scan": None
        }
        return empty_manifest
    
    # Download and parse manifest
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp_file:
            local_temp = tmp_file.name
        
        scp_from_remote(f"~/{manifest_path}", local_temp, "shr-zion.stanford.edu", USER, PASSWORD)
        
        with open(local_temp, 'r') as f:
            manifest = json.load(f)
        
        os.unlink(local_temp)
        return manifest
        
    except Exception as e:
        print(f"⚠️  Error reading job manifest: {e}")
        return {"jobs": {}, "last_scan": None}


def _save_job_manifest(sshProc, manifest):
    """Save the job manifest to remote server."""
    manifest_path = _get_manifest_path()
    
    try:
        import tempfile
        from datetime import datetime
        
        # Update last_scan timestamp
        manifest["last_scan"] = datetime.now().isoformat()
        
        # Write to local temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(manifest, tmp_file, indent=2)
            local_temp = tmp_file.name
        
        # Upload to remote
        scp_to_remote(local_temp, f"~/{manifest_path}", "shr-zion.stanford.edu", USER, PASSWORD)
        
        os.unlink(local_temp)
        
    except Exception as e:
        print(f"⚠️  Error saving job manifest: {e}")


def _register_job(sshProc, job_id, operation_type, status="running"):
    """Register a new job or operation in the manifest."""
    from datetime import datetime
    
    manifest = _load_job_manifest(sshProc)
    
    # Initialize job entry if it doesn't exist
    if job_id not in manifest["jobs"]:
        manifest["jobs"][job_id] = {
            "job_id": job_id,
            "operations": {},
            "created_by": USER,
            "last_updated": datetime.now().isoformat()
        }
    
    # Register the operation
    manifest["jobs"][job_id]["operations"][operation_type] = {
        "status": status,
        "started_at": datetime.now().isoformat(),
        "completed_at": None
    }
    manifest["jobs"][job_id]["last_updated"] = datetime.now().isoformat()
    
    _save_job_manifest(sshProc, manifest)


def _update_job_status(sshProc, job_id, operation_type, status):
    """Update the status of a job operation."""
    from datetime import datetime
    
    manifest = _load_job_manifest(sshProc)
    
    if job_id in manifest["jobs"] and operation_type in manifest["jobs"][job_id]["operations"]:
        manifest["jobs"][job_id]["operations"][operation_type]["status"] = status
        if status == "completed":
            manifest["jobs"][job_id]["operations"][operation_type]["completed_at"] = datetime.now().isoformat()
        manifest["jobs"][job_id]["last_updated"] = datetime.now().isoformat()
        
        _save_job_manifest(sshProc, manifest)


def _scan_and_update_jobs(sshProc, verbose=False):
    """Scan existing job directories and update manifest with current status."""
    central_tmp_dir = "beak_tmp"
    
    # Get list of project directories
    success, dir_list = execute_remote_command(sshProc, f"ls -1 ~/{central_tmp_dir}/ 2>/dev/null | grep '^beak_' || echo 'no_jobs'", timeout=10)
    
    if "no_jobs" in dir_list:
        return {"jobs": {}, "last_scan": None}
    
    manifest = _load_job_manifest(sshProc)
    project_dirs = [d.strip() for d in dir_list.strip().split('\n') if d.strip() and d.strip().startswith('beak_')]
    
    for project_dir in project_dirs:
        job_id = project_dir
        
        if verbose:
            print(f"   Scanning job: {job_id}")
        
        # Initialize job if not in manifest
        if job_id not in manifest["jobs"]:
            from datetime import datetime
            manifest["jobs"][job_id] = {
                "job_id": job_id,
                "operations": {},
                "created_by": USER,
                "last_updated": datetime.now().isoformat()
            }
        
        # Check each operation type
        operations_to_check = ["search", "align", "taxonomy"]
        for op_type in operations_to_check:
            op_dir = f"{central_tmp_dir}/{project_dir}/{op_type}"
            
            # Check if operation directory exists
            success, op_exists = execute_remote_command(sshProc, f"test -d ~/{op_dir} && echo 'exists' || echo 'not_exists'", timeout=5)
            
            if "exists" in op_exists:
                # Determine status based on completion files
                if op_type == "search":
                    success, complete_check = execute_remote_command(sshProc, f"test -f ~/{op_dir}/resultDB && echo 'complete' || echo 'incomplete'", timeout=5)
                elif op_type == "align":
                    success, complete_check = execute_remote_command(sshProc, f"test -f ~/{op_dir}/aligned.fasta && echo 'complete' || echo 'incomplete'", timeout=5)
                elif op_type == "taxonomy":
                    success, complete_check = execute_remote_command(sshProc, f"test -f ~/{op_dir}/taxonomyResult && echo 'complete' || echo 'incomplete'", timeout=5)
                
                status = "completed" if "complete" in complete_check else "running"
                
                # Update manifest
                if op_type not in manifest["jobs"][job_id]["operations"]:
                    from datetime import datetime
                    manifest["jobs"][job_id]["operations"][op_type] = {
                        "status": status,
                        "started_at": datetime.now().isoformat(),
                        "completed_at": datetime.now().isoformat() if status == "completed" else None
                    }
                else:
                    # Update existing operation status
                    old_status = manifest["jobs"][job_id]["operations"][op_type]["status"]
                    if old_status != status:
                        manifest["jobs"][job_id]["operations"][op_type]["status"] = status
                        if status == "completed" and not manifest["jobs"][job_id]["operations"][op_type]["completed_at"]:
                            from datetime import datetime
                            manifest["jobs"][job_id]["operations"][op_type]["completed_at"] = datetime.now().isoformat()
    
    _save_job_manifest(sshProc, manifest)
    return manifest


def _get_project_structure(project_id, operation_type):
    """
    Create project directory structure with operation-specific subdirectories.
    
    Args:
        project_id (str): The project identifier
        operation_type (str): 'search', 'align', or 'taxonomy'
        
    Returns:
        dict: Directory paths for the project structure
    """
    base_dir = "beak_tmp"
    project_dir = f"{base_dir}/{project_id}"
    operation_dir = f"{project_dir}/{operation_type}"
    
    return {
        "base_dir": base_dir,
        "project_dir": project_dir, 
        "operation_dir": operation_dir,
        "config_file": f"{operation_dir}/config.json",
        "log_file": f"{operation_dir}/job.log"
    }


def _clean_terminal_output(output):
    """Clean terminal escape sequences from command output and extract relevant results"""
    if not output:
        return ""
    
    # Remove escape sequences, carriage returns, and ANSI codes
    cleaned = output.strip()
    cleaned = cleaned.replace('\x1b[?2004l', '').replace('\x1b[?2004h', '').replace('\r', '')
    
    # Remove BEAK_END markers that contaminate output
    import re
    beak_marker_pattern = re.compile(r'BEAK_END_\d+_\d+')
    cleaned = beak_marker_pattern.sub('', cleaned)
    
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cleaned = ansi_escape.sub('', cleaned)
    
    # Split by lines to find the actual command output
    lines = cleaned.split('\n')
    
    # Look for our expected command outputs (these should be at the end after banner)
    expected_patterns = [
        'search', 'align', 'no', 'exists', 'not_found', 'complete', 'incomplete', 
        'no_process', 'found_process', 'mmseqs_running', 'mmseqs_found', 'no_mmseqs',
        'no_pgrep', 'lock_exists', 'no_lock', 'pid_running', 'pid_dead', 'no_pid',
        'SEARCH_SUCCESS', 'SEARCH_FAILED', 'no_status', 'ssh_test_ok', 'no_jobs', 
        'no_log', 'no_config', 'conda_not_found', 'beak_not_found', 'DIFF_COMPLETE',
        'no_projects'
    ]
    
    # Check last few lines for expected patterns (command output usually comes last)
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
            
        # If line exactly matches expected pattern, return it
        if line in expected_patterns:
            return line
            
        # If line contains expected pattern, return the pattern
        for pattern in expected_patterns:
            if pattern in line:
                return pattern
        
        # Special case: look for 'exists' or 'missing' anywhere in the line
        if 'exists' in line.lower() and 'missing' not in line.lower():
            return 'exists'
        elif 'missing' in line.lower() and 'exists' not in line.lower():
            return 'missing'
                
        # If line contains process info (ps output), return the whole line
        if 'mmseqs' in line.lower() and ('python' in line or 'search' in line):
            return line
            
        # Special handling for conda/brew paths
        if '/conda' in line or '/brew' in line:
            return line
            
        # Special handling for beak project directories
        if line.startswith('beak_') and not any(bad in line.lower() for bad in ['error', 'not found', 'failed']):
            return line
    
    # Remove system banners and login messages
    filtered_lines = []
    skip_patterns = [
        'Documentation:', 'Management:', 'Support:', 'System information',
        'System load:', 'Usage of /', 'Memory usage:', 'Swap usage:', 'Temperature:',
        'IPv4 address', 'Strictly confined', 'Expanded Security Maintenance',
        'updates can be applied', 'To see these additional', 'Learn more about',
        'The list of available updates', 'To check for new updates',
        'New release', 'Run \'do-release-upgrade\'', 'Last login:'
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip system banner lines
        if any(pattern in line for pattern in skip_patterns):
            continue
        # Skip lines that are just asterisks or dashes
        if re.match(r'^[\*\-\=\s]+$', line):
            continue
        filtered_lines.append(line)
    
    # If we have filtered lines, return them joined
    if filtered_lines:
        return '\n'.join(filtered_lines)
    
    # Last resort: scan the entire output for expected simple patterns
    full_text = cleaned.lower()
    if 'exists' in full_text and 'missing' not in full_text:
        return 'exists'
    elif 'missing' in full_text and 'exists' not in full_text:
        return 'missing'
    elif 'complete' in full_text and 'incomplete' not in full_text:
        return 'complete'
    elif 'incomplete' in full_text:
        return 'incomplete'
    
    # If no expected patterns found, return the basic cleaned output
    return cleaned

def _get_local_project_dir(project_id, local_results_dir=None):
    """
    Get the local project directory path with consistent default handling.
    
    Args:
        project_id (str): The project/job ID
        local_results_dir (str, optional): Custom local directory. Defaults to "my_beak_projects"
        
    Returns:
        str: Full path to local project directory
    """
    if local_results_dir is None:
        local_results_dir = "my_beak_projects"
    return f"{local_results_dir}/{project_id}"

def list_local_projects(local_results_dir=None, verbose=True):
    """
    List all local BEAK projects and their contents.
    
    Args:
        local_results_dir (str, optional): Directory to scan. Defaults to "my_beak_projects"
        verbose (bool): Show detailed file information
        
    Returns:
        dict: Project information with file counts and sizes
    """
    import os
    import glob
    from datetime import datetime
    
    if local_results_dir is None:
        local_results_dir = "my_beak_projects"
    
    if not os.path.exists(local_results_dir):
        if verbose:
            print(f"📁 No local projects directory found at: {local_results_dir}")
            print(f"   Projects will be created here when you first retrieve results")
        return {}
    
    projects = {}
    project_dirs = glob.glob(f"{local_results_dir}/beak_*")
    
    if verbose:
        print(f"📁 Local BEAK projects in {local_results_dir}/:")
        if not project_dirs:
            print("   No projects found")
            return projects
    
    for project_path in sorted(project_dirs):
        project_id = os.path.basename(project_path)
        
        # Get basic info
        try:
            stat_info = os.stat(project_path)
            modified_time = datetime.fromtimestamp(stat_info.st_mtime)
            
            # Count files and get total size
            files = glob.glob(f"{project_path}/*")
            file_count = len(files)
            total_size = sum(os.path.getsize(f) for f in files if os.path.isfile(f))
            
            # Determine project type based on files
            project_types = []
            if any("search_results" in f for f in files):
                project_types.append("search")
            if any("aligned" in f for f in files):
                project_types.append("align")
            if any("tree" in f for f in files):
                project_types.append("tree")
            if any("taxonomy" in f for f in files):
                project_types.append("taxonomy")
            
            project_info = {
                "path": project_path,
                "files": file_count,
                "size_mb": round(total_size / (1024*1024), 2),
                "modified": modified_time,
                "types": project_types
            }
            
            projects[project_id] = project_info
            
            if verbose:
                types_str = ", ".join(project_types) if project_types else "unknown"
                print(f"   📊 {project_id}")
                print(f"      Types: {types_str} | Files: {file_count} | Size: {project_info['size_mb']} MB")
                print(f"      Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Error reading {project_id}: {e}")
    
    return projects

def _get_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return key

def authenticate(user=None, host=None):
    global USER, PASSWORD
    
    # If no user provided, try to use the previously stored one
    if user is None:
        if USER is not None:
            user = USER
            print(f"🔄 Using previously authenticated user: {user}")
        else:
            raise ValueError("No user provided and no previous authentication found. Please provide a username.")
    
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

def get_current_user():
    """Get the currently authenticated username."""
    return USER

def is_authenticated():
    """Check if a user is currently authenticated."""
    return USER is not None and get_pw() is not None

def clear_authentication():
    """Clear the current authentication session."""
    global USER, PASSWORD
    USER = None
    PASSWORD = None
    print("🔓 Authentication cleared")

def show_auth_status():
    """Display current authentication status."""
    if USER is not None:
        has_stored_pw = get_pw() is not None
        print(f"🔑 Currently authenticated as: {USER}")
        print(f"📁 Password stored: {'Yes' if has_stored_pw else 'No'}")
        if has_stored_pw:
            print("✅ Ready to connect without re-entering credentials")
        else:
            print("⚠️  May need to re-enter password on next connection")
    else:
        print("🔓 No user currently authenticated")
        print("💡 Run authenticate('your_username') to log in")

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

    # Try to use existing authentication first
    if USER is not None:
        try:
            stored_password = get_pw()
            if stored_password is not None:
                PASSWORD = stored_password
                print(f"🔑 Using stored credentials for: {USER}")
            else:
                print(f"🔄 Re-authenticating user: {USER}")
                authenticate(USER)
        except Exception as e:
            print(f"⚠️  Error retrieving stored credentials: {e}")
            print("🔐 Please re-authenticate")
            user = input("Username: ")
            authenticate(user)
    else:
        user = input("Username: ")
        authenticate(user)

    # Find sshpass executable path
    sshpass_path = shutil.which('sshpass')
    if sshpass_path is None:
        raise RuntimeError("sshpass not found in PATH. Please install sshpass.")

    sshProc = subprocess.Popen(
        [sshpass_path, '-p', PASSWORD, 'ssh', '-tt', f'{USER}@{HOST}'],
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
    import random
    marker = f"BEAK_END_{int(time.time())}_{random.randint(1000, 9999)}"
    full_cmd = f"{cmd} && echo '{marker}' || echo '{marker}'"
    
    sshProc.stdin.write(full_cmd + '\n')
    sshProc.stdin.flush()
    
    out = ''
    start_time = time.time()
    marker_found = False
    
    # Read all available output until we see our marker or timeout
    while True:
        rlist, _, _ = select.select([sshProc.stdout], [], [], 0.5)
        if rlist:
            line = sshProc.stdout.readline()
            if not line:
                break
            
            # Check if we found our end marker
            if marker in line:
                marker_found = True
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
    
    # Remove shell prompt patterns and command echoes
    lines = cleaned_out.split('\n')
    clean_lines = []
    
    # Common system banner patterns to skip
    banner_patterns = [
        'Documentation:', 'Management:', 'Support:', 'System information',
        'System load:', 'Usage of /', 'Memory usage:', 'Swap usage:', 'Temperature:',
        'IPv4 address', 'Processes:', 'Users logged in:', 'Strictly confined',
        'Expanded Security Maintenance', 'updates can be applied', 'To see these additional',
        'Learn more about', 'New release', 'do-release-upgrade', 'Last login:',
        '*** System restart required ***', 'https://', 'Run \'do-release-upgrade\''
    ]
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, shell prompts, command echoes, and system banners
        if (line and 
            not line.startswith('(base)') and 
            not line.startswith('$') and 
            not line.startswith('#') and
            line != cmd.strip() and
            not any(pattern in line for pattern in banner_patterns) and
            ('echo' not in line.lower() or 'echo' in cmd.lower())):
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
    
    # Simple success check - if we got output and no obvious errors, consider it successful
    success = result is not None and not any(indicator in result.lower() for indicator in ['command not found', 'no such file', 'permission denied'])
    
    return success, result

def check_remote_process(sshProc, process_patterns, job_id=None, verbose=False, debug=False):
    """
    Check if specific processes are running on the remote server using ps command.
    
    Args:
        sshProc: SSH process connection
        process_patterns (list): List of process name patterns to search for (e.g., ['mmseqs', 'clustalo', 'iqtree'])
        job_id (str, optional): Job ID to filter processes for specific project
        verbose (bool): Show detailed output
        debug (bool): Show debug information including raw ps output
        
    Returns:
        dict: {
            'running_processes': [list of matching process info],
            'is_running': bool,
            'process_count': int,
            'patterns_found': [list of patterns that matched]
        }
    """
    if verbose:
        print(f"🔍 Checking for running processes: {process_patterns}")
    
    # Construct ps command to look for processes
    # Use full format to get more information about processes
    ps_cmd = "ps aux --no-headers"
    
    if debug:
        print(f"🔧 DEBUG CMD: {ps_cmd}")
    
    success, ps_output = execute_remote_command(sshProc, ps_cmd, timeout=15, error_reporting=debug)
    
    if not success or not ps_output:
        if verbose:
            print("⚠️ Could not retrieve process list")
        return {
            'running_processes': [],
            'is_running': False,
            'process_count': 0,
            'patterns_found': []
        }
    
    if debug:
        print(f"🔧 DEBUG: Raw ps output length: {len(ps_output)} chars")
        print(f"🔧 DEBUG: First 300 chars of ps output: {ps_output[:300]}...")
    
    # Clean the output and parse processes
    cleaned_output = _clean_terminal_output(ps_output)
    if not cleaned_output:
        cleaned_output = ps_output  # Fallback to raw output if cleaning failed
    
    lines = cleaned_output.split('\n')
    running_processes = []
    patterns_found = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check each pattern against the process line
        for pattern in process_patterns:
            if pattern.lower() in line.lower():
                # If job_id is specified, also check if it's related to this job
                if job_id:
                    if job_id in line:
                        running_processes.append(line)
                        if pattern not in patterns_found:
                            patterns_found.append(pattern)
                        if debug:
                            print(f"🔧 DEBUG: Found job-specific process: {line}")
                else:
                    running_processes.append(line)
                    if pattern not in patterns_found:
                        patterns_found.append(pattern)
                    if debug:
                        print(f"🔧 DEBUG: Found matching process: {line}")
                break  # Don't match multiple patterns for the same line
    
    result = {
        'running_processes': running_processes,
        'is_running': len(running_processes) > 0,
        'process_count': len(running_processes),
        'patterns_found': patterns_found
    }
    
    if verbose:
        if result['is_running']:
            print(f"✅ Found {result['process_count']} running processes matching: {patterns_found}")
            for proc in running_processes[:3]:  # Show first 3 processes
                print(f"   📋 {proc}")
            if len(running_processes) > 3:
                print(f"   ... and {len(running_processes) - 3} more")
        else:
            print(f"❌ No running processes found for patterns: {process_patterns}")
    
    return result

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

def make_temp_dir(sshProc, remote_dir="beak_tmp"):
    """Ensure the remote beak_tmp directory exists.
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

def nest(verbose=False):
    """ Set up home environment on remote server
    
    Args:
        verbose (bool): If True, show detailed output during setup
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
    print("==== 1. Conda ====")
    
    # Simple check - try to run conda --version
    success, conda_version = execute_remote_command(sshProc, "conda --version 2>/dev/null || echo 'CONDA_NOT_FOUND'", timeout=10)
    
    if success and "CONDA_NOT_FOUND" not in conda_version:
        conda_found = True
        # Get the conda path
        success_path, conda_path_result = execute_remote_command(sshProc, "which conda 2>/dev/null || echo 'CONDA_PATH_NOT_FOUND'", timeout=5)
        if "CONDA_PATH_NOT_FOUND" not in conda_path_result:
            print(f"✅ Conda found: {conda_version.strip()} at {conda_path_result.strip()}")
        else:
            print(f"✅ Conda found: {conda_version.strip()}")
    else:
        conda_found = False
        print("❌ Conda not found")
    
    if not conda_found:
        print("❌ Conda is not installed on the remote server.")
        print("🔄 Installing Miniconda on the remote server...")
        install_cmd = (
            "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && "
            "bash ~/miniconda.sh -b -p $HOME/miniconda && "
            "rm ~/miniconda.sh && "
            "echo 'export PATH=\"$HOME/miniconda/bin:$PATH\"' >> ~/.bashrc"
        )
        success, install_result = execute_remote_command(sshProc, install_cmd, timeout=300)
        if success:
            print("✅ Miniconda installation complete.")
        else:
            print("⚠️  Miniconda installation may have issues, but continuing...")
    print()
    time.sleep(0.5)

    # Now check for beak conda environment
    print("\n==== 2. Check beak Conda Environment ====")
    
    if not conda_found:
        print("❌ Cannot check conda environments - conda not available")
        env_exists = False
    else:
        # Simple direct check for beak environment
        success, env_check = execute_remote_command(sshProc, "conda info --envs | grep -w beak || echo 'BEAK_ENV_NOT_FOUND'", timeout=15)
        
        if success and "BEAK_ENV_NOT_FOUND" not in env_check:
            env_exists = True
            # Extract the path from the output
            env_line = env_check.strip()
            if env_line:
                parts = env_line.split()
                if len(parts) >= 2:
                    env_path = parts[-1]  # Last part is usually the path
                    print(f"✅ Found beak environment at: {env_path}")
                else:
                    print("✅ beak environment exists")
            else:
                print("✅ beak environment exists")
        else:
            env_exists = False
            print("❌ beak environment not found")
            
            # Show available environments for debugging
            success_list, env_list = execute_remote_command(sshProc, "conda info --envs | grep -v '^#' | awk '{print $1}' | grep -v '^$' | head -5", timeout=10)
            if success_list and env_list.strip():
                available_envs = [env.strip() for env in env_list.split('\n') if env.strip()]
                if available_envs:
                    print(f"   Available environments: {', '.join(available_envs)}")

    if not env_exists:
        print("🔄 Creating beak environment...")
    else:
        print("✅ beak environment already exists, skipping creation")
        
        # Create beak_tmp directory first
        print("   🗂️  Creating beak_tmp directory...")
        success, mkdir_result = execute_remote_command(sshProc, "mkdir -p ~/beak_tmp", timeout=10, error_reporting=False)
        if not success:
            # Try alternative approach
            success, mkdir_result2 = execute_remote_command(sshProc, "test -d ~/beak_tmp || mkdir ~/beak_tmp", timeout=10, error_reporting=False)
            if not success:
                print(f"⚠️  beak_tmp directory creation uncertain, but continuing...")
        else:
            print("   ✅ beak_tmp directory ready")
        
        # Find the beak_env.yml file - it should be in the same directory as this utils.py file
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        beak_env_path = os.path.join(current_dir, "beak_env.yml")
        
        if not os.path.exists(beak_env_path):
            print(f"❌ Error: beak_env.yml not found at {beak_env_path}")
            print("Please ensure beak_env.yml exists in the src/beak/remote/ directory")
            return False
            
        print("   📤 Uploading beak_env.yml...")
        try:
            scp_to_remote(beak_env_path, "~/beak_tmp/beak_env.yml", 'shr-zion.stanford.edu', USER, PASSWORD)
            print("✅ beak_env.yml uploaded successfully")
        except Exception as e:
            print(f"❌ Failed to upload beak_env.yml: {e}")
            print("   This may be due to SSH connection issues or file permissions")
            return False
            
        print("   🔄 Creating conda environment (this may take a few minutes)...")
        create_env_cmd = f"conda env create -n beak -f ~/beak_tmp/beak_env.yml 2>/dev/null || conda env update -n beak -f ~/beak_tmp/beak_env.yml"
        success, create_result = execute_remote_command(sshProc, create_env_cmd, timeout=300, error_reporting=False)
        if success or "already exists" in create_result.lower():
            print("✅ beak environment created successfully!")
        else:
            print(f"⚠️  Environment creation may have issues but continuing...")
            if "already exists" in create_result.lower():
                print("   (Environment already exists - this is OK)")
    print()
    time.sleep(0.5)

    # 3. Check packages in beak environment against beak_env.yml
    print("\n==== 3. Package Consistency Check ====")
    
    # Upload beak_env.yml if it doesn't exist yet
    if not env_exists:
        # The beak_env.yml should already be uploaded to beak_tmp from step 2
        print("✅ beak_env.yml uploaded to beak_tmp successfully")
    
    # Simple package count check instead of full diff
    if env_exists:
        print("🔍 Checking environment packages...")
        success, pkg_check = execute_remote_command(sshProc, "conda list -n beak 2>/dev/null | grep -c '^[a-zA-Z]' || echo '0'", timeout=15)
        pkg_check_clean = _clean_terminal_output(pkg_check)
        
        try:
            pkg_count = int(pkg_check_clean.strip())
            if pkg_count > 0:
                print(f"✅ beak environment has {pkg_count} packages installed")
            else:
                print("⚠️  beak environment appears to be empty or not accessible")
        except ValueError:
            print("⚠️  Could not determine package count")
    else:
        print("✅ beak environment will be created with required packages")
    print()
    time.sleep(0.5)

    # Check for Homebrew
    print("\n==== 4. Checking for Homebrew ====")
    # Simple check - try to run brew --version
    success, brew_version = execute_remote_command(sshProc, "brew --version 2>/dev/null | head -1 || echo 'BREW_NOT_FOUND'", timeout=10)
    
    if success and "BREW_NOT_FOUND" not in brew_version:
        # Get the brew path
        success_path, brew_path_result = execute_remote_command(sshProc, "which brew 2>/dev/null || echo 'BREW_PATH_NOT_FOUND'", timeout=5)
        if "BREW_PATH_NOT_FOUND" not in brew_path_result:
            print(f"✅ Homebrew found: {brew_version.strip()} at {brew_path_result.strip()}")
        else:
            print(f"✅ Homebrew found: {brew_version.strip()}")
    else:
        # Try common brew locations (Linux and macOS)
        brew_locations = [
            "/home/linuxbrew/.linuxbrew/bin/brew",  # Linux Homebrew
            "~/.linuxbrew/bin/brew",                # User Linuxbrew
            "/opt/homebrew/bin/brew",               # macOS ARM
            "/usr/local/bin/brew"                   # macOS Intel
        ]
        
        brew_found = False
        brew_path = None
        
        for location in brew_locations:
            success_alt, check_alt = execute_remote_command(sshProc, f"test -f {location} && echo '{location}' || echo 'not_found'", timeout=5)
            check_alt_clean = _clean_terminal_output(check_alt)
            if "not_found" not in check_alt_clean and location in check_alt_clean:
                brew_found = True
                brew_path = check_alt_clean.strip()
                break
        
        if brew_found:
            print(f"✅ Homebrew found at: {brew_path}")
        else:
            print("❌ Homebrew not found in common locations")
            print("🔄 Installing Homebrew for Linux (this may take several minutes)...")
            
            # Check if we're on Linux and install appropriate version
            success_os, os_check = execute_remote_command(sshProc, "uname -s", timeout=5)
            os_type = _clean_terminal_output(os_check).strip().lower() if success_os else "linux"
            
            if "linux" in os_type:
                # Install Homebrew for Linux (Linuxbrew)
                install_brew_cmd = 'NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                success, install_result = execute_remote_command(sshProc, install_brew_cmd, timeout=600, error_reporting=False)
                
                if success:
                    # Add Homebrew to PATH for Linux
                    print("   🔧 Setting up Homebrew environment...")
                    setup_cmds = [
                        'echo "eval \"\$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)\"" >> ~/.bashrc',
                        'echo "eval \"\$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)\"" >> ~/.profile'
                    ]
                    for cmd in setup_cmds:
                        execute_remote_command(sshProc, cmd, timeout=10, error_reporting=False)
                    
                    print("✅ Homebrew installation complete.")
                    print("   📝 Note: You may need to restart your shell or run 'source ~/.bashrc'")
                else:
                    print("⚠️  Homebrew installation may have issues, but continuing...")
                    if "already installed" in install_result.lower():
                        print("   (Homebrew may already be installed - this is OK)")
            else:
                print(f"⚠️  Detected OS: {os_type}. Using generic Homebrew installation...")
                install_brew_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                success, install_result = execute_remote_command(sshProc, install_brew_cmd, timeout=600, error_reporting=False)
                
                if success:
                    print("✅ Homebrew installation complete.")
                else:
                    print("⚠️  Homebrew installation may have issues, but continuing...")
    print()
    time.sleep(1)

    # Check for mmseqs (placeholder for future implementation)
    print("\n==== 5. Additional Tools ====")
    print("✅ Additional tool checks can be added here in the future.")
    print()

    print("\n🎉 Nesting complete! Your remote environment is ready.")
    print("✅ nest() completed successfully!")
    
    # Properly close SSH connection
    try:
        sshProc.stdin.close()
        sshProc.stdout.close()
        sshProc.terminate()
        sshProc.wait(timeout=5)  # Wait for clean termination
    except Exception as e:
        if verbose:
            print(f"Warning: Error closing SSH connection: {e}")
    
    return True
    

def _find_existing_search(sshProc, search_hash, db, query_content, verbose=False):
    """
    Helper function to find existing searches with the same parameters.
    
    Args:
        sshProc: SSH process
        search_hash (str): Hash of search parameters
        db (str): Database name
        query_content (str): Query sequence content
        verbose (bool): Show detailed output
        
    Returns:
        dict: Existing job info if found, None otherwise
    """
    import json
    
    # List all beak job directories
    list_cmd = "ls -1 ~/beak_tmp/ 2>/dev/null | grep '^beak_' || echo 'no_jobs'"
    job_dirs = ssend(list_cmd, sshProc, timeout=10).strip()
    
    if job_dirs == 'no_jobs' or not job_dirs:
        return None
        
    # Check each job directory for matching config
    for job_dir in job_dirs.split('\n'):
        job_dir = job_dir.strip()
        if not job_dir or not job_dir.startswith('beak_'):
            continue
            
        config_path = f"~/beak_tmp/{job_dir}/search/config.json"
        
        # Check if config exists and read it
        check_config_cmd = f"test -f {config_path} && cat {config_path} || echo 'no_config'"
        config_content = ssend(check_config_cmd, sshProc, timeout=10).strip()
        
        if config_content == 'no_config' or not config_content:
            continue
            
        try:
            config_data = json.loads(config_content)
            
            # Check if this is a search job with matching parameters
            if (config_data.get('database') == db and 
                config_data.get('query_sequence') == query_content):
                
                # Verify resultDB exists to ensure search completed successfully  
                resultdb_path = f"~/beak_tmp/{job_dir}/search/resultDB"
                check_resultdb_cmd = f"test -d {resultdb_path} && echo 'exists' || echo 'missing'"
                resultdb_status = ssend(check_resultdb_cmd, sshProc, timeout=10).strip()
                
                if resultdb_status == 'exists':
                    if verbose:
                        print(f"   ✅ Found completed search: {job_dir}")
                        print(f"   📊 Database: {config_data.get('database')}")
                        print(f"   📅 Created: {config_data.get('timestamp', 'unknown')}")
                    
                    return {
                        "job_id": job_dir,
                        "status": "found",
                        "config": config_path
                    }
                elif verbose:
                    print(f"   ⚠️ Found incomplete search: {job_dir} (missing resultDB)")
                    
        except json.JSONDecodeError:
            if verbose:
                print(f"   ⚠️ Invalid config in {job_dir}")
            continue
    
    return None


def _get_available_databases(sshProc, verbose=False):
    """
    Dynamically detect available databases on the remote server.
    Uses caching to avoid repeated server checks.
    
    Args:
        sshProc: SSH connection to the remote server
        verbose (bool): If True, show detailed output
        
    Returns:
        dict: Mapping of database names to their paths/identifiers
    """
    global _DATABASE_CACHE, _DATABASE_CACHE_TIMESTAMP
    
    # Check if we have a valid cache
    current_time = time.time()
    if (_DATABASE_CACHE and _DATABASE_CACHE_TIMESTAMP and 
        current_time - _DATABASE_CACHE_TIMESTAMP < _DATABASE_CACHE_TIMEOUT):
        if verbose:
            print("🗄️  Using cached database list")
        return _DATABASE_CACHE
    
    if verbose:
        print("🔍 Scanning remote server for available databases...")
    
    db_dir = "/srv/protein_sequence_databases"
    
    # Get directory listing with better error handling
    success, ls_result = execute_remote_command(sshProc, f"ls -1 {db_dir}/ 2>/dev/null || echo 'no_access'", timeout=10)
    
    # Clean the output to remove any SSH banners or escape sequences
    ls_result_clean = _clean_terminal_output(ls_result) if ls_result else ""
    
    if verbose:
        print(f"   Raw ls result: '{ls_result[:200] if ls_result else 'None'}...'")
        print(f"   Cleaned ls result: '{ls_result_clean[:200] if ls_result_clean else 'None'}...'")
    
    if not success or "no_access" in ls_result_clean or not ls_result_clean.strip():
        if verbose:
            print(f"⚠️  Could not access {db_dir}, using fallback database list")
        # Fallback to basic databases with common name variations
        return {
            'UniRef90': 'UniRef90',
            'uniref90': 'UniRef90', 
            'UniRef50': 'UniRef50',
            'uniref50': 'UniRef50',
            'swissprot': 'swissprot',
            'SwissProt': 'swissprot',
        }
    
    files = [f.strip() for f in ls_result_clean.strip().split('\n') if f.strip()]
    
    if verbose:
        print(f"   Found {len(files)} files/directories in database directory")
    
    # Build database mapping based on what's available
    available_dbs = {}
    
    # Check for mmseqs databases (files without extensions that have corresponding .dbtype files)
    for file in files:
        if file.endswith('.dbtype'):
            db_name = file.replace('.dbtype', '')
            # Skip internal mmseqs files but include main databases
            if not db_name.endswith('_h') and not db_name.startswith('_') and db_name not in ['mapping', 'taxonomy']:
                available_dbs[db_name] = db_name
                
                # Add case-insensitive mappings for common databases
                db_lower = db_name.lower()
                if db_lower != db_name:
                    available_dbs[db_lower] = db_name
                
                # Add specific mappings for common variations
                if db_name == 'UniRef90':
                    available_dbs['uniref90'] = db_name
                elif db_name == 'UniRef50':  
                    available_dbs['uniref50'] = db_name
                elif db_name == 'swissprot':
                    available_dbs['SwissProt'] = db_name
    
    # Check for FASTA files that could be used
    fasta_files = [f for f in files if f.endswith('.fa') or f.endswith('.fasta')]
    for fasta_file in fasta_files:
        base_name = fasta_file.replace('.fa', '').replace('.fasta', '')
        # Create friendly names for FASTA databases
        if 'uniprot_all' in base_name:
            available_dbs['uniprot_all'] = fasta_file
        elif 'uniref90' in base_name.lower():
            available_dbs['uniref90_fasta'] = fasta_file
        elif 'bfd' in base_name:
            available_dbs['bfd'] = fasta_file
        elif 'mgy_clusters' in base_name:
            available_dbs['mgy_clusters'] = fasta_file
        elif 'pdb_seqres' in base_name:
            available_dbs['pdb_seqres'] = fasta_file
        elif 'nt_rna' in base_name:
            available_dbs['nt_rna'] = fasta_file
        elif 'rfam' in base_name:
            available_dbs['rfam'] = fasta_file
        elif 'rnacentral' in base_name:
            available_dbs['rnacentral'] = fasta_file
    
    # Cache the results
    _DATABASE_CACHE = available_dbs
    _DATABASE_CACHE_TIMESTAMP = time.time()
    
    if verbose:
        print(f"🗄️  Found {len(available_dbs)} available databases:")
        for name, path in sorted(available_dbs.items()):
            print(f"   {name}: {path}")
    
    # If no databases found, provide a sensible fallback
    if not available_dbs:
        if verbose:
            print("⚠️  No databases detected, using fallback list")
        available_dbs = {
            'UniRef90': 'UniRef90',
            'uniref90': 'UniRef90', 
            'UniRef50': 'UniRef50',
            'uniref50': 'UniRef50',
            'swissprot': 'swissprot',
            'SwissProt': 'swissprot',
        }
    
    return available_dbs


def list_databases(sshProc=None, verbose=True):
    """
    List all available databases on the remote server.
    
    Args:
        sshProc: SSH connection to the remote server
        verbose (bool): If True, show detailed information about each database
        
    Returns:
        dict: Mapping of database names to their paths/identifiers
    """
    if sshProc is None:
        sshProc = sopen()
    
    available_dbs = _get_available_databases(sshProc, verbose)
    
    if not verbose:
        return available_dbs
    
    print("\n📋 Available Databases:")
    print("=" * 50)
    
    # Group databases by type
    mmseqs_dbs = {}
    fasta_dbs = {}
    
    for name, path in available_dbs.items():
        if path.endswith('.fa') or path.endswith('.fasta'):
            fasta_dbs[name] = path
        else:
            mmseqs_dbs[name] = path
    
    if mmseqs_dbs:
        print("\n🗄️  MMseqs2 Databases (optimized, fastest):")
        for name, path in sorted(mmseqs_dbs.items()):
            print(f"   {name:20} → {path}")
    
    if fasta_dbs:
        print("\n📄 FASTA Databases (converted on-the-fly):")
        for name, path in sorted(fasta_dbs.items()):
            print(f"   {name:20} → {path}")
    
    print(f"\n💡 Usage: beak.remote.search(sequence, db='database_name')")
    print(f"🚀 Recommended: Use MMseqs2 databases for best performance")
    
    return available_dbs


def search(query, db="UniRef90", sshProc=None, verbose=False, force_new=False, user_id=None, job_id=None, sensitivity=3.0):
    """
    Start mmseqs2 search to find similar protein sequences in protein databases.
    Automatically checks for existing identical searches to avoid duplicates.
    This function starts the search and returns immediately with job info.
    Use status() to check job progress and retrieve_results() to get completed results.
    
    Args:
        query (str): Either a protein sequence string or a local FASTA file path.
        db (str): Database name. Available databases are detected dynamically from the server.
                 Use list_databases() to see available options. Default: "UniRef90".
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed output. If False, show minimal progress.
        force_new (bool): If True, skip duplicate checking and force new search.
        user_id (str, optional): Custom job identifier (deprecated, use job_id instead).
        job_id (str, optional): Custom job identifier. If None, generates gibberish words.
        sensitivity (float): Search sensitivity from 1.0 (fastest) to 7.0 (most sensitive). Default: 3.0.
    Returns:
        dict: {"job_id": "unique_id", "status": "started/found", "config": "config_file.json"} or None on failure.
    """
    import uuid
    import time
    import hashlib
    import json
    
    if sshProc is None:
        sshProc = sopen()
    
    # Validate sensitivity parameter
    if not isinstance(sensitivity, (int, float)):
        raise ValueError(f"Sensitivity must be a number, got {type(sensitivity).__name__}")
    
    sensitivity = float(sensitivity)
    if not (1.0 <= sensitivity <= 7.0):
        raise ValueError(f"Sensitivity must be between 1.0 and 7.0, got {sensitivity}")
    
    if verbose:
        print(f"🔧 Using sensitivity: {sensitivity}")
    
    # Create a hash of the search parameters for deduplication
    if os.path.isfile(query):
        with open(query, 'r') as f:
            query_content = f.read().strip()
    else:
        query_content = query.strip()
    
    search_params = {
        "query_sequence": query_content,
        "database": db,
        "mmseqs_command": "mmseqs search",
        "settings": {
            "sensitivity": sensitivity,
            "sort_results": 1
        }
    }
    
    # Create hash for deduplication
    search_hash = hashlib.md5(json.dumps(search_params, sort_keys=True).encode()).hexdigest()[:12]
    
    central_tmp_dir = "beak_tmp"
    
    # Check for existing searches unless force_new is True
    if not force_new:
        print("🔍 Checking for existing identical searches...")
        existing_job = _find_existing_search(sshProc, search_hash, db, query_content, verbose)
        if existing_job:
            print(f"✅ Found existing search: {existing_job['job_id']}")
            return existing_job
    
    # Generate unique project directory for this search - use new project-based structure
    # Prefer job_id parameter over user_id for backward compatibility
    custom_id = job_id or user_id
    project_id = _generate_project_id(custom_id)
    
    # Get project structure with search subdirectory
    structure = _get_project_structure(project_id, "search")
    project_dir = structure["project_dir"]
    search_dir = structure["operation_dir"]
    
    # Dynamically check available databases on the remote server
    if verbose:
        print("🔍 Checking available databases...")
    available_dbs = _get_available_databases(sshProc, verbose)
    
    if db not in available_dbs:
        print(f"❌ Database '{db}' not available")
        print(f"📋 Available databases: {', '.join(sorted(available_dbs.keys()))}")
        raise ValueError(f"Database '{db}' not available. Choose from: {list(sorted(available_dbs.keys()))}")
    
    db_path = f"/srv/protein_sequence_databases/{available_dbs[db]}"
    
    if sshProc is None:
        sshProc = sopen()

    print(f"🔍 Starting mmseqs sequence search with database: {db}")
    if verbose:
        print(f"📁 Creating project directory: {project_id}")
        print(f"📂 Search operation directory: {search_dir}")
    
    # Create project directory structure with search subdirectory
    ssend(f"mkdir -p ~/{project_dir}", sshProc, timeout=10)
    ssend(f"mkdir -p ~/{search_dir}", sshProc, timeout=10)
    
    # Step 1: Create query FASTA file
    print("📝 Creating query FASTA file...")
    remote_query_fasta = f"~/{search_dir}/query.fasta"
    
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
        "operation_type": "search",
        "timestamp": datetime.now().isoformat(),
        "query_type": "file" if os.path.isfile(query) else "sequence",
        "query_sequence": query_sequence if not os.path.isfile(query) else f"File: {query}",
        "sequence_length": len(query_sequence) if not os.path.isfile(query) else "N/A (from file)",
        "database": db,
        "database_path": db_path,
        "project_directory": project_dir,
        "search_directory": search_dir,
        "mmseqs_command": "mmseqs search",
        "settings": {
            "sensitivity": sensitivity,
            "sort_results": 1,
            "createdb_timeout": 10,
            "search_timeout": 1800,
            "convert_timeout": 30
        }
    }
    
    config_json = json.dumps(config_data, indent=2)
    remote_config = structure["config_file"]
    
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
            print(f"   ✅ Search config saved to: {remote_config}")
            print(f"      (Operation-specific config in search/ subdirectory)")
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
    create_db_cmd = f"cd ~/{search_dir} && {mmseqs_path} createdb query.fasta queryDB && echo 'DB_CREATION_COMPLETE'"
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
        success, verify_result = execute_remote_command(sshProc, f"test -f ~/{search_dir}/queryDB && echo 'exists' || echo 'missing'", timeout=5)
        
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
        f"cd ~/{search_dir} && {mmseqs_path} search queryDB {db_path} "
        f"resultDB tmp -s {sensitivity} --sort-results 1"
    )
    if verbose:
        print(f"   🔧 Command: {search_cmd}")
    
    # Capture the search output
    if verbose:
        print("   🚀 Starting search and capturing output...")
    
    # Run the search command with output redirection
    # Create a comprehensive background process with tracking
    background_script = f"""
cd ~/{search_dir} && \\
echo "SEARCH_STARTED" > .running && \\
echo "$$" > .pid && \\
({search_cmd} > search_output.log 2>&1; EXIT_CODE=$?; \\
 if [ $EXIT_CODE -eq 0 ]; then echo "SEARCH_SUCCESS" > .status; else echo "SEARCH_FAILED" > .status; fi; \\
 rm -f .running .pid; exit $EXIT_CODE) &
"""
    
    if verbose:
        print(f"   📝 Starting tracked background process...")
        print(f"   📤 Process will create lock file (.running) and PID file (.pid)")
    
    # Start the tracked background search
    ssend(background_script.strip(), sshProc, timeout=5)
    
    # Wait a moment for the process to start
    import time
    time.sleep(3)
    
    # Check if there are any immediate errors in the log
    if verbose:
        print("   🔍 Checking for immediate errors...")
        success, log_check = execute_remote_command(sshProc, f"head -10 ~/{search_dir}/search_output.log 2>/dev/null || echo 'no_log_yet'", timeout=5, error_reporting=True)
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
        success, error_check = execute_remote_command(sshProc, f"cat ~/{search_dir}/search_output.log 2>/dev/null || echo 'no_log'", timeout=5, error_reporting=verbose)
        if success and "no_log" not in error_check:
            print(f"   ❌ Error in log: {error_check}")
    
    # Verify the search started successfully
    time.sleep(2)  # Give process time to start
    success, process_info = execute_remote_command(sshProc, "ps aux | grep mmseqs | grep -v grep", timeout=5, error_reporting=verbose)
    if success and process_info.strip():
        print("   ✅ mmseqs search started successfully")
        if verbose:
            print(f"   🔍 Process info: {process_info.strip()}")
        
        # Download config file immediately to local project directory
        local_project_dir = _get_local_project_dir(project_id)
        os.makedirs(local_project_dir, exist_ok=True)
        local_config = f"{local_project_dir}/search_config.json"
        try:
            scp_from_remote(remote_config, local_config, "shr-zion.stanford.edu", USER, PASSWORD)
            if verbose:
                print(f"✅ Config saved to: {local_config}")
        except Exception as e:
            print(f"⚠️  Warning: Failed to download config: {e}")
            local_config = None
        
        print(f"🚀 Search job '{project_id}' started on remote server")
        print(f"   Use status('{project_id}') to check progress")
        print(f"   Use retrieve_results('{project_id}') when complete")
        
        # Register job in manifest
        _register_job(sshProc, project_id, "search", "running")
        
        return {
            "job_id": project_id,
            "project_id": project_id,
            "status": "running", 
            "config": local_config,
            "remote_dir": project_dir,
            "search_dir": search_dir
        }
    else:
        print("   ❌ mmseqs process not found - search may have failed to start")
        # Check if there's an error message in the log
        success, error_check = execute_remote_command(sshProc, f"cat ~/{search_dir}/search_output.log 2>/dev/null || echo 'no_log'", timeout=5, error_reporting=verbose)
        if success and "no_log" not in error_check:
            print(f"   ❌ Error in log: {error_check}")
        return None


def status_old(job_id=None, sshProc=None, verbose=False):
    """
    Check the status of running or completed beak jobs (search/align).
    
    Args:
        job_id (str, optional): Specific job ID to check. If None, shows all jobs.
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed output.
    Returns:
        dict: Status information for the job(s).
    """
    if sshProc is None:
        sshProc = sopen()
    
    central_tmp_dir = "beak_tmp"
    
    if job_id:
        # Check specific job
        project_dir = f"{central_tmp_dir}/{job_id}"
        
        # Check if project directory exists
        success, dir_check = execute_remote_command(sshProc, f"test -d ~/{project_dir} && echo 'exists' || echo 'not_found'", timeout=5)
        dir_check_clean = _clean_terminal_output(dir_check)
        if verbose:
            print(f"   🔍 Directory check raw: '{dir_check[:100]}...'")  # Show first 100 chars for debugging
            print(f"   🔍 Directory check clean: '{dir_check_clean}'")
        if "not_found" in dir_check_clean or (not dir_check_clean and "not_found" in dir_check):
            print(f"❌ Job '{job_id}' not found")
            return {"job_id": job_id, "status": "not_found"}
        
        # Determine operations by checking what subdirectories and configs exist
        success, check_search = execute_remote_command(sshProc, f"if [ -f ~/{project_dir}/search/config.json ]; then echo 'search'; else echo 'no'; fi", timeout=10)
        success, check_align = execute_remote_command(sshProc, f"if [ -f ~/{project_dir}/align/config.json ]; then echo 'align'; else echo 'no'; fi", timeout=10)
        success, check_taxonomy = execute_remote_command(sshProc, f"if [ -f ~/{project_dir}/taxonomy/config.json ]; then echo 'taxonomy'; else echo 'no'; fi", timeout=10)
        
        check_search_clean = _clean_terminal_output(check_search)
        check_align_clean = _clean_terminal_output(check_align)
        check_taxonomy_clean = _clean_terminal_output(check_taxonomy)
        
        if verbose:
            print(f"   🔍 Search config raw: '{check_search[:50]}...'")
            print(f"   🔍 Search config check: '{check_search_clean}'")
            print(f"   🔍 Align file raw: '{check_align[:50]}...'")
            print(f"   🔍 Align file check: '{check_align_clean}'")
            print(f"   🔍 Taxonomy config raw: '{check_taxonomy[:50]}...'")
            print(f"   🔍 Taxonomy config check: '{check_taxonomy_clean}'")
        
        if "search" in check_search_clean:
            job_type = "search"
        elif "align" in check_align_clean:
            job_type = "align"
        elif "taxonomy" in check_taxonomy_clean:
            job_type = "taxonomy"
        else:
            # Default to search if we can't determine (likely a new/running search job)
            job_type = "search"
            
        if verbose:
            print(f"   🎯 Detected job type: {job_type}")
        
        if job_type == "search":
            search_dir = f"{project_dir}/search"
            
            # First, test basic SSH connectivity
            success_test, ssh_test = execute_remote_command(sshProc, "echo 'ssh_test_ok'", timeout=5)
            
            # Check for mmseqs process - look for processes using files in our project directory
            success, process_check = execute_remote_command(sshProc, f"ps aux | grep mmseqs | grep {job_id}; if [ $? -eq 0 ]; then echo 'found_process'; else echo 'no_process'; fi", timeout=10)
            
            # Also check for any mmseqs process and get details
            success_simple, process_simple = execute_remote_command(sshProc, f"ps aux | grep mmseqs | grep -v grep; if [ $? -eq 0 ]; then echo 'mmseqs_found'; else echo 'no_mmseqs'; fi", timeout=10)
            
            # Check for lock file or running indicator
            success_lock, lock_check = execute_remote_command(sshProc, f"if [ -f ~/{search_dir}/.running ]; then echo 'lock_exists'; else echo 'no_lock'; fi", timeout=10)
            
            # Check for PID file and verify process is actually running
            success_pid, pid_check = execute_remote_command(sshProc, f"if [ -f ~/{search_dir}/.pid ]; then PID=$(cat ~/{search_dir}/.pid); if kill -0 $PID 2>/dev/null; then echo 'pid_running'; else echo 'pid_dead'; fi; else echo 'no_pid'; fi", timeout=10)
            
            # Check for status file
            success_status, status_check = execute_remote_command(sshProc, f"if [ -f ~/{search_dir}/.status ]; then cat ~/{search_dir}/.status; else echo 'no_status'; fi", timeout=10)
            
            # Check for search completion
            success2, result_check = execute_remote_command(sshProc, f"if [ -f ~/{search_dir}/resultDB ]; then echo 'complete'; else echo 'incomplete'; fi", timeout=10)
        elif job_type == "taxonomy":
            taxonomy_dir = f"{project_dir}/taxonomy"
            
            # Check for mmseqs taxonomy process - similar to search but look for taxonomy
            success, process_check = execute_remote_command(sshProc, f"ps aux | grep mmseqs | grep taxonomy | grep {job_id}; if [ $? -eq 0 ]; then echo 'found_process'; else echo 'no_process'; fi", timeout=10)
            
            # Also check for any mmseqs taxonomy process
            success_simple, process_simple = execute_remote_command(sshProc, f"ps aux | grep 'mmseqs taxonomy' | grep -v grep; if [ $? -eq 0 ]; then echo 'mmseqs_found'; else echo 'no_mmseqs'; fi", timeout=10)
            
            # Check for lock file or running indicator
            success_lock, lock_check = execute_remote_command(sshProc, f"if [ -f ~/{taxonomy_dir}/.running ]; then echo 'lock_exists'; else echo 'no_lock'; fi", timeout=10)
            
            # Check for PID file and verify process is actually running
            success_pid, pid_check = execute_remote_command(sshProc, f"if [ -f ~/{taxonomy_dir}/.pid ]; then PID=$(cat ~/{taxonomy_dir}/.pid); if kill -0 $PID 2>/dev/null; then echo 'pid_running'; else echo 'pid_dead'; fi; else echo 'no_pid'; fi", timeout=10)
            
            # Check for status file
            success_status, status_check = execute_remote_command(sshProc, f"if [ -f ~/{taxonomy_dir}/.status ]; then cat ~/{taxonomy_dir}/.status; else echo 'no_status'; fi", timeout=10)
            
            # Check for taxonomy completion
            success2, result_check = execute_remote_command(sshProc, f"if [ -f ~/{taxonomy_dir}/taxonomyResult ]; then echo 'complete'; else echo 'incomplete'; fi", timeout=10)
        else:
            # Alignment jobs - check for PID file and process, plus status file
            align_dir = f"{project_dir}/align"
            
            # Check for PID file and verify process is actually running
            success_pid, pid_check = execute_remote_command(sshProc, f"if [ -f ~/{align_dir}/.pid ]; then PID=$(cat ~/{align_dir}/.pid); if kill -0 $PID 2>/dev/null; then echo 'pid_running'; else echo 'pid_dead'; fi; else echo 'no_pid'; fi", timeout=10)
            
            # Check for general clustalo process (fallback)
            success, process_check = execute_remote_command(sshProc, f"ps aux | grep -E '(clustalo|awk)' | grep {job_id} | grep -v grep; if [ $? -eq 0 ]; then echo 'found_process'; else echo 'no_process'; fi", timeout=10)
            
            # Check for status file
            success_status, status_check = execute_remote_command(sshProc, f"if [ -f ~/{align_dir}/.status ]; then cat ~/{align_dir}/.status; else echo 'no_status'; fi", timeout=10)
            
            # Check for alignment completion
            success2, result_check = execute_remote_command(sshProc, f"if [ -f ~/{align_dir}/aligned.fasta ]; then echo 'complete'; else echo 'incomplete'; fi", timeout=10)
        
        # Clean up escape sequences and check for actual process
        process_check_clean = _clean_terminal_output(process_check)
        result_check_clean = _clean_terminal_output(result_check)
        
        # Clean additional checks for search, taxonomy, and alignment jobs
        if job_type in ["search", "taxonomy"]:
            if job_type == "search":
                ssh_test_clean = _clean_terminal_output(ssh_test)
            process_simple_clean = _clean_terminal_output(process_simple)
            lock_check_clean = _clean_terminal_output(lock_check)  
            pid_check_clean = _clean_terminal_output(pid_check)
            status_check_clean = _clean_terminal_output(status_check)
        else:
            # Alignment jobs
            pid_check_clean = _clean_terminal_output(pid_check)
            status_check_clean = _clean_terminal_output(status_check)
        
        if verbose:
            print(f"   🔍 Job type: {job_type}")
            if job_type == "search":
                print(f"   🔍 SSH Test: '{ssh_test_clean}' (success: {success_test})")
            print(f"   🔍 Process check raw (full): '{process_check}'")
            print(f"   🔍 Process check clean: '{process_check_clean}'")
            if job_type in ["search", "taxonomy"]:
                print(f"   🔍 Simple process check raw (full): '{process_simple}'")
                print(f"   🔍 Simple process check clean: '{process_simple_clean}'")
                print(f"   🔍 Lock check: '{lock_check_clean}'")
            
            # PID and status checks for all job types that support them
            if job_type in ["search", "taxonomy", "align"]:
                print(f"   🔍 PID check: '{pid_check_clean}'")
                print(f"   🔍 Status check: '{status_check_clean}'")
            
            print(f"   🔍 Result check raw (full): '{result_check}'") 
            print(f"   🔍 Result check clean: '{result_check_clean}'")
        
        # Check if process is running - use multiple detection methods based on job type
        if job_type in ["search", "taxonomy"]:
            is_running = (success and "found_process" in process_check_clean) or \
                        (success_simple and "mmseqs_found" in process_simple_clean) or \
                        (success_lock and "lock_exists" in lock_check_clean) or \
                        (success_pid and "pid_running" in pid_check_clean)
        else:
            # Alignment jobs - check PID first, then fallback to process search
            is_running = (success_pid and "pid_running" in pid_check_clean) or \
                        (success and "found_process" in process_check_clean)
        
        # Check completion - use multiple methods for alignment jobs
        if job_type == "align":
            is_complete = ("complete" in result_check_clean) or \
                         (success_status and "completed" in status_check_clean)
        else:
            is_complete = "complete" in result_check_clean
        
        if verbose:
            print(f"   🎯 Process analysis:")
            if job_type in ["search", "taxonomy"]:
                if job_type == "search":
                    print(f"      - SSH test success: {success_test}")
                print(f"      - Directory-based process check: '{process_check_clean}' (success: {success})")
                print(f"      - General mmseqs check: '{process_simple_clean}' (success: {success_simple})")
                print(f"      - Lock file check: '{lock_check_clean}' (success: {success_lock})")
                print(f"      - PID check: '{pid_check_clean}' (success: {success_pid})")
                print(f"      - Status file check: '{status_check_clean}' (success: {success_status})")
                
                dir_found = success and "found_process" in process_check_clean
                general_found = success_simple and "mmseqs_found" in process_simple_clean
                lock_found = success_lock and "lock_exists" in lock_check_clean
                pid_found = success_pid and "pid_running" in pid_check_clean
                
                print(f"      - Directory-based detection: {dir_found}")
                print(f"      - General mmseqs detection: {general_found}")
                print(f"      - Lock file detection: {lock_found}")
                print(f"      - PID-based detection: {pid_found}")
            else:
                # Alignment jobs
                print(f"      - Process check: '{process_check_clean}' (success: {success})")
                print(f"      - PID check: '{pid_check_clean}' (success: {success_pid})")
                print(f"      - Status file check: '{status_check_clean}' (success: {success_status})")
                
                pid_found = success_pid and "pid_running" in pid_check_clean
                process_found = success and "found_process" in process_check_clean
                
                print(f"      - PID-based detection: {pid_found}")
                print(f"      - Process-based detection: {process_found}")
            print(f"      - Final is_running: {is_running}")
            print(f"   🎯 Result analysis:")
            print(f"      - result_check_clean: '{result_check_clean}'")
            print(f"      - 'complete' in clean: {'complete' in result_check_clean}")
            print(f"      - Final is_complete: {is_complete}")
        
        # Check log file for progress/errors
        if job_type == "search":
            log_path = f"~/{project_dir}/search/search_output.log"
        elif job_type == "taxonomy":
            log_path = f"~/{project_dir}/taxonomy/taxonomy_output.log"
        else:
            log_path = f"~/{project_dir}/align/job.log"
        success, log_content = execute_remote_command(sshProc, f"tail -10 {log_path} 2>/dev/null || echo 'no_log'", timeout=10)
        # Clean up escape sequences from log content
        log_content_clean = _clean_terminal_output(log_content)
        
        # Determine status
        if is_complete:
            status_str = "completed"
            print(f"✅ {job_type.title()} job '{job_id}' is completed")
            if job_type == "search":
                print(f"   Use retrieve_results('{job_id}') to download results")
            elif job_type == "align":
                print(f"   Use retrieve_results('{job_id}') to download alignment")
                print(f"   Remote file: ~/{project_dir}/align/aligned.fasta")
            elif job_type == "taxonomy":
                print(f"   Taxonomy results: ~/{project_dir}/taxonomy/taxonomyResult")
            else:
                print(f"   Results in: ~/{project_dir}/")
        elif is_running:
            status_str = "running"
            print(f"🔄 {job_type.title()} job '{job_id}' is still running")
            if verbose and log_content_clean and "no_log" not in log_content_clean:
                print(f"   Recent output: {log_content_clean}")
        else:
            # Not running and not complete - check log for errors
            status_str = "failed"
            print(f"❌ {job_type.title()} job '{job_id}' appears to have failed")
            if log_content_clean and "no_log" not in log_content_clean:
                print(f"   Log output: {log_content_clean}")
        
        return {
            "job_id": job_id,
            "job_type": job_type,
            "status": status_str,
            "is_running": is_running,
            "is_complete": is_complete,
            "remote_dir": project_dir,
            "log_content": log_content_clean if "no_log" not in log_content_clean else None
        }
    
    else:
        # List all jobs
        print("📋 Checking all beak jobs...")
        
        # List all project directories (unified beak jobs)
        success, dir_list = execute_remote_command(sshProc, f"ls -1 ~/{central_tmp_dir}/ 2>/dev/null | grep '^beak_' || echo 'no_jobs'", timeout=10)
        
        if "no_jobs" in dir_list or not dir_list.strip():
            print("   No jobs found")
            return {"jobs": []}
        
        jobs = []
        project_dirs = [d.strip() for d in dir_list.strip().split('\n') if d.strip() and d.strip().startswith('beak_')]
        
        for project_dir in project_dirs:
            job_id = project_dir  # job_id is the same as directory name
            
            # Determine job type by checking what files exist
            success, check_search = execute_remote_command(sshProc, f"test -f ~/{central_tmp_dir}/{project_dir}/search/config.json && echo 'search' || echo 'no'", timeout=5)
            success, check_align = execute_remote_command(sshProc, f"test -f ~/{central_tmp_dir}/{project_dir}/align/aligned.fasta && echo 'align' || echo 'no'", timeout=5)
            success, check_taxonomy = execute_remote_command(sshProc, f"test -f ~/{central_tmp_dir}/{project_dir}/taxonomy/config.json && echo 'taxonomy' || echo 'no'", timeout=5)
            
            if "search" in check_search:
                job_type = "search"
            elif "align" in check_align:
                job_type = "align"
            elif "taxonomy" in check_taxonomy:
                job_type = "taxonomy"
            else:
                # Default to search if we can't determine
                job_type = "search"
            
            # Check for different process types based on job type
            if job_type == "search":
                # Check for mmseqs process
                success, process_check = execute_remote_command(sshProc, f"ps aux | grep mmseqs | grep {job_id} | grep -v grep || echo 'no_process'", timeout=5)
                # Check for search completion
                success2, result_check = execute_remote_command(sshProc, f"test -f ~/{central_tmp_dir}/{project_dir}/search/resultDB && echo 'complete' || echo 'incomplete'", timeout=5)
            elif job_type == "taxonomy":
                # Check for mmseqs taxonomy process
                success, process_check = execute_remote_command(sshProc, f"ps aux | grep 'mmseqs taxonomy' | grep {job_id} | grep -v grep || echo 'no_process'", timeout=5)
                # Check for taxonomy completion
                success2, result_check = execute_remote_command(sshProc, f"test -f ~/{central_tmp_dir}/{project_dir}/taxonomy/taxonomyResult && echo 'complete' || echo 'incomplete'", timeout=5)
            else:
                # Check for clustalo or awk process (alignment jobs)
                success, process_check = execute_remote_command(sshProc, f"ps aux | grep -E '(clustalo|awk)' | grep {job_id} | grep -v grep || echo 'no_process'", timeout=5)
                # Check for alignment completion
                success2, result_check = execute_remote_command(sshProc, f"test -f ~/{central_tmp_dir}/{project_dir}/align/aligned.fasta && echo 'complete' || echo 'incomplete'", timeout=5)
            
            process_check_clean = _clean_terminal_output(process_check)
            result_check_clean = _clean_terminal_output(result_check)
            
            is_running = success and process_check_clean and "no_process" not in process_check_clean and len(process_check_clean) > 0
            is_complete = "complete" in result_check_clean
            
            if is_complete:
                status_str = "completed ✅"
            elif is_running:
                status_str = "running 🔄"
            else:
                status_str = "failed ❌"
            
            print(f"   {job_id} ({job_type}): {status_str}")
            jobs.append({
                "job_id": job_id,
                "job_type": job_type,
                "status": status_str.split()[0],  # Just the status word
                "is_running": is_running,
                "is_complete": is_complete
            })
        
        return {"jobs": jobs}


def retrieve_results(job_id, sshProc=None, verbose=False, debug=False, wait_for_completion=True, max_wait_minutes=30, local_results_dir=None):
    """
    Retrieve results from a completed job (search, alignment, or tree).
    
    Args:
        job_id (str): Job ID of the completed job.
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed output.
        debug (bool): If True, show debug information.
        wait_for_completion (bool): If True, wait for processes to complete before retrieving files.
        max_wait_minutes (int): Maximum time to wait for completion.
        local_results_dir (str, optional): Local directory to store results. Defaults to "my_beak_projects".
    Returns:
        dict: Job results based on type:
              Search: {"results": "file.tsv", "fasta": "file.fasta", "config": "config.json", "job_type": "search"}
              Alignment: {"aligned_fasta": "file.fasta", "job_id": "id", "job_type": "align"}
              Tree: {"tree_file": "file.treefile", "log_file": "file.log", "job_id": "id", "job_type": "tree"}
    """
    import time
    import json
    
    if sshProc is None:
        sshProc = sopen()
    
    # Use my_beak_projects as default to keep repo clean
    if local_results_dir is None:
        local_results_dir = "my_beak_projects"
    
    # Create local project directory using utility function
    local_project_dir = _get_local_project_dir(job_id, local_results_dir)
    os.makedirs(local_project_dir, exist_ok=True)
    if verbose:
        print(f"📁 Created local directory: {local_project_dir}")
    
    central_tmp_dir = "beak_tmp"
    project_dir = f"{central_tmp_dir}/{job_id}"
    
    # Determine job type by checking what operation subdirectories exist
    if debug:
        print(f"🔧 DEBUG: Checking project directory: ~/{project_dir}")
    
    success, check_search = execute_remote_command(sshProc, f"test -f ~/{project_dir}/search/config.json && echo 'search' || echo 'no'", timeout=10, error_reporting=debug)
    success, check_align = execute_remote_command(sshProc, f"test -f ~/{project_dir}/align/config.json && echo 'align' || echo 'no'", timeout=10, error_reporting=debug)
    success, check_tree = execute_remote_command(sshProc, f"test -f ~/{project_dir}/tree/config.json && echo 'tree' || echo 'no'", timeout=10, error_reporting=debug)
    success, check_taxonomy = execute_remote_command(sshProc, f"test -f ~/{project_dir}/taxonomy/config.json && echo 'taxonomy' || echo 'no'", timeout=10, error_reporting=debug)
    
    check_search_clean = _clean_terminal_output(check_search)
    check_align_clean = _clean_terminal_output(check_align)
    check_tree_clean = _clean_terminal_output(check_tree)
    check_taxonomy_clean = _clean_terminal_output(check_taxonomy)
    
    if verbose:
        print(f"   🔍 Search config check: '{check_search_clean}'")
        print(f"   🔍 Align config check: '{check_align_clean}'")
        print(f"   🔍 Tree config check: '{check_tree_clean}'")
        print(f"   🔍 Taxonomy config check: '{check_taxonomy_clean}'")
    
    # Determine job type (prioritize most recent operations)
    if "tree" in check_tree_clean:
        job_type = "tree"
        operation_dir = f"{project_dir}/tree"
    elif "align" in check_align_clean:
        job_type = "align"
        operation_dir = f"{project_dir}/align"
    elif "search" in check_search_clean:
        job_type = "search"
        operation_dir = f"{project_dir}/search"
    elif "taxonomy" in check_taxonomy_clean:
        job_type = "taxonomy"
        operation_dir = f"{project_dir}/taxonomy"
    else:
        print(f"❌ No valid job configuration found for '{job_id}'")
        print(f"   Checked: ~/{project_dir}/{{search,align,tree,taxonomy}}/config.json")
        return None
    
    print(f"📥 Retrieving {job_type} results for job '{job_id}'...")
    
    # Read config file to get settings and db_path information
    config_path = f"~/{operation_dir}/config.json"
    success, config_content = execute_remote_command(sshProc, f"cat {config_path} 2>/dev/null", timeout=10, error_reporting=debug)
    
    config_data = {}
    if success and config_content:
        try:
            config_cleaned = _clean_terminal_output(config_content)
            if config_cleaned:
                config_data = json.loads(config_cleaned)
                if verbose:
                    print(f"✅ Loaded job configuration from {config_path}")
                    if debug:
                        print(f"🔧 DEBUG: Config data keys: {list(config_data.keys())}")
        except json.JSONDecodeError as e:
            if debug:
                print(f"🔧 DEBUG: Failed to parse config JSON: {e}")
                print(f"🔧 DEBUG: Raw config content: {config_content[:200]}...")
    
    # Check if processes are still running before retrieving files
    if wait_for_completion:
        process_patterns = []
        if job_type == "search":
            process_patterns = ["mmseqs"]
        elif job_type == "align":
            process_patterns = ["clustalo", "clustalw"]
        elif job_type == "tree":
            process_patterns = ["iqtree", "iqtree2"]
        elif job_type == "taxonomy":
            process_patterns = ["mmseqs"]
        
        if process_patterns:
            print(f"🔍 Checking if {job_type} processes are still running...")
            process_check = check_remote_process(sshProc, process_patterns, job_id=job_id, verbose=verbose, debug=debug)
            
            if process_check['is_running']:
                print(f"⏳ Found {process_check['process_count']} running {job_type} processes")
                print(f"   Patterns found: {process_check['patterns_found']}")
                
                if wait_for_completion:
                    print(f"⏳ Waiting for processes to complete (max {max_wait_minutes} minutes)...")
                    wait_start = time.time()
                    wait_interval = 30  # Check every 30 seconds
                    
                    while process_check['is_running'] and (time.time() - wait_start) < (max_wait_minutes * 60):
                        time.sleep(wait_interval)
                        process_check = check_remote_process(sshProc, process_patterns, job_id=job_id, verbose=False, debug=False)
                        
                        elapsed_minutes = (time.time() - wait_start) / 60
                        if process_check['is_running']:
                            print(f"⏳ Still running... ({elapsed_minutes:.1f}/{max_wait_minutes} minutes elapsed)")
                        else:
                            print(f"✅ Processes completed after {elapsed_minutes:.1f} minutes")
                            break
                    
                    if process_check['is_running']:
                        print(f"⚠️ Processes still running after {max_wait_minutes} minutes")
                        print("   Proceeding with file retrieval - some files may be incomplete")
                else:
                    print("   Proceeding with retrieval despite running processes...")
            else:
                if verbose:
                    print(f"✅ No active {job_type} processes found - ready for file retrieval")
    
    # Check if job is complete based on job type and expected output files
    expected_files = []
    if job_type == "search":
        expected_files = [f"~/{operation_dir}/resultDB"]
    elif job_type == "align":
        expected_files = [f"~/{operation_dir}/aligned.fasta"]
    elif job_type == "tree":
        expected_files = [f"~/{operation_dir}/alignment.fasta.treefile"]
    elif job_type == "taxonomy":
        expected_files = [f"~/{operation_dir}/taxonomyResult"]
    
    # Check if all expected files exist
    missing_files = []
    for file_path in expected_files:
        success, file_check = execute_remote_command(sshProc, f"test -f {file_path} && echo 'exists' || echo 'missing'", timeout=10, error_reporting=debug)
        file_check_clean = _clean_terminal_output(file_check)
        
        if debug:
            print(f"🔧 DEBUG: File check for {file_path}: '{file_check_clean}'")
        
        if "missing" in file_check_clean:
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ {job_type.title()} job '{job_id}' is not yet complete")
        print(f"   Missing files: {missing_files}")
        if wait_for_completion:
            print(f"   (Process checking was enabled - files may still be generating)")
        print(f"   Use status('{job_id}') to check progress")
        return None
    
    if verbose:
        print(f"✅ All expected {job_type} output files found")
    
    if job_type == "search":
        # Handle search job results - use config data for db_path if available
        db_path = config_data.get("database_path", "/srv/protein_sequence_databases/UniRef90")
        if verbose:
            print(f"📄 Using database path from config: {db_path}")
        
        # Find mmseqs path (use config or fallback detection)
        mmseqs_path = config_data.get("mmseqs_path", "mmseqs")
        if mmseqs_path == "mmseqs":  # Need to find actual path
            success, mmseqs_check = execute_remote_command(sshProc, "which mmseqs", timeout=10, error_reporting=debug)
            mmseqs_check_clean = _clean_terminal_output(mmseqs_check)
            
            if debug:
                print(f"🔧 DEBUG: mmseqs check: '{mmseqs_check_clean}'")
            
            if success and mmseqs_check_clean.strip() and "not found" not in mmseqs_check_clean.lower():
                mmseqs_path = mmseqs_check_clean.strip().split('\n')[-1].strip()
            else:
                # Try common locations
                for location in ["/usr/local/bin/mmseqs", "/opt/mmseqs/bin/mmseqs", "~/bin/mmseqs"]:
                    success, test_result = execute_remote_command(sshProc, f"test -f {location} && echo 'found'", timeout=5)
                    if success and "found" in _clean_terminal_output(test_result):
                        mmseqs_path = location
                        break
                else:
                    print("❌ mmseqs not found. Cannot convert results.")
                    return None
        
        # Convert results to readable format
        print("📋 Converting results to readable format...")
        convert_cmd = f"cd ~/{operation_dir} && {mmseqs_path} createtsv queryDB resultDB search_results.tsv"
        success, convert_result = execute_remote_command(sshProc, convert_cmd, timeout=60, error_reporting=debug)
        
        if not success:
            print(f"❌ Error converting results: {convert_result}")
            return None
        
        print("   ✅ Results converted to TSV format")
        
        # Generate FASTA file from search results
        print("🧬 Generating FASTA file from search results...")
        
        # Step 1: Create sequence file database using config db_path
        if verbose:
            print("   🗄️  Creating sequence file database...")
        createseqdb_cmd = f"cd ~/{operation_dir} && {mmseqs_path} createseqfiledb {db_path} resultDB fastaDB"
        success, createseqdb_result = execute_remote_command(sshProc, createseqdb_cmd, timeout=120, error_reporting=debug)
        
        if not success:
            print(f"❌ Error creating sequence file database: {createseqdb_result}")
            return None
        
        if verbose:
            print("   ✅ Sequence file database created")
        
        # Step 2: Convert to FASTA format using result2flat
        if verbose:
            print("   📄 Converting to FASTA format...")
        result2flat_cmd = f"cd ~/{search_dir} && {mmseqs_path} result2flat {db_path} {db_path} fastaDB search_results.fasta --use-fasta-header TRUE"
        success, result2flat_result = execute_remote_command(sshProc, result2flat_cmd, timeout=120, error_reporting=verbose)
        
        if not success:
            print(f"❌ Error converting to FASTA: {result2flat_result}")
            return None
        
        print("   ✅ FASTA file generated from search results")
        
        # Check what files actually exist on the remote server
        if verbose:
            print("   🔍 Checking available files on remote server...")
            success, file_list = execute_remote_command(sshProc, f"ls -la ~/{search_dir}/", timeout=10, error_reporting=verbose)
            if success:
                print(f"   📋 Remote files: {_clean_terminal_output(file_list)}")
        
        # Download results, FASTA, and config files
        remote_results_file = f"~/{search_dir}/search_results.tsv"
        remote_fasta_file = f"~/{search_dir}/search_results.fasta"
        remote_config = f"~/{search_dir}/config.json"
        local_result = f"{local_project_dir}/search_results.tsv"
        local_fasta = f"{local_project_dir}/search_results.fasta"
        local_config = f"{local_project_dir}/search_config.json"
        
        try:
            # Download TSV results file
            if verbose:
                print(f"   📥 Downloading TSV from: {remote_results_file}")
            scp_from_remote(remote_results_file, local_result, "shr-zion.stanford.edu", USER, PASSWORD)
            print(f"✅ TSV results saved to: {local_result}")
            
            # Download FASTA results file - check if it exists first
            if verbose:
                print(f"   📥 Checking FASTA file: {remote_fasta_file}")
            success, fasta_check = execute_remote_command(sshProc, f"test -f {remote_fasta_file} && echo 'exists' || echo 'missing'", timeout=5)
            fasta_exists = "exists" in _clean_terminal_output(fasta_check)
            
            if fasta_exists:
                if verbose:
                    print(f"   📥 Downloading FASTA from: {remote_fasta_file}")
                scp_from_remote(remote_fasta_file, local_fasta, "shr-zion.stanford.edu", USER, PASSWORD)
                print(f"✅ FASTA results saved to: {local_fasta}")
            else:
                print(f"⚠️  FASTA file not found at: {remote_fasta_file}")
                # Try to regenerate it
                if verbose:
                    print("   🔄 Attempting to regenerate FASTA file...")
                local_fasta = None
            
            # Download config file
            if verbose:
                print(f"   📥 Downloading config from: {remote_config}")
            scp_from_remote(remote_config, local_config, "shr-zion.stanford.edu", USER, PASSWORD)
            if verbose:
                print(f"✅ Config saved to: {local_config}")
            
        except Exception as e:
            print(f"❌ Failed to retrieve files: {e}")
            if verbose:
                print(f"   You can manually retrieve results from: {remote_results_file}")
                print(f"   You can manually retrieve FASTA from: {remote_fasta_file}")
                print(f"   You can manually retrieve config from: {remote_config}")
            return None
        
        if verbose:
            print(f"🧹 Remote files remain in: {project_dir} (for debugging)")
        
        # Update job status to completed in manifest
        _update_job_status(sshProc, job_id, "search", "completed")
        
        return {
            "job_type": "search",
            "results": local_result, 
            "fasta": local_fasta,
            "config": local_config
        }
    
    elif job_type == "taxonomy":
        # Handle taxonomy job results
        print("📋 Converting taxonomy results to TSV format...")
        
        # Find mmseqs path (reuse logic from search function)
        success, mmseqs_check = execute_remote_command(sshProc, "which mmseqs", timeout=10, error_reporting=verbose)
        mmseqs_check_clean = _clean_terminal_output(mmseqs_check)
        
        if not success or not mmseqs_check_clean.strip() or "not found" in mmseqs_check_clean.lower():
            # Try common locations
            mmseqs_path = None
            for location in ["/usr/local/bin/mmseqs", "/opt/mmseqs/bin/mmseqs", "~/bin/mmseqs"]:
                success, test_result = execute_remote_command(sshProc, f"test -f {location} && echo 'found'", timeout=5)
                if success and "found" in test_result:
                    mmseqs_path = location
                    break
            if not mmseqs_path:
                print("❌ mmseqs not found. Cannot convert taxonomy results.")
                return None
        else:
            mmseqs_path_line = mmseqs_check_clean.strip().split('\n')[-1]
            if mmseqs_path_line and '/' in mmseqs_path_line:
                mmseqs_path = mmseqs_path_line.strip()
            else:
                mmseqs_path = "mmseqs"
        
        # Read config to get the sequence database used
        success, config_content = execute_remote_command(sshProc, f"cat ~/{taxonomy_dir}/config.json", timeout=10, error_reporting=verbose)
        config_content_clean = _clean_terminal_output(config_content)
        
        seqdb_path = None
        if success and config_content_clean:
            try:
                import json
                config_data = json.loads(config_content_clean)
                input_type = config_data.get("input_type", "unknown")
                parent_project_id = config_data.get("parent_project_id")
                
                if parent_project_id:
                    # Use queryDB from parent search job
                    seqdb_path = f"~/{central_tmp_dir}/{parent_project_id}/search/queryDB"
                    if verbose:
                        print(f"   📄 Using queryDB from parent project: {parent_project_id}")
                elif input_type in ["file", "content"]:
                    # Use seqDB from current taxonomy directory (created from FASTA)
                    seqdb_path = f"~/{taxonomy_dir}/seqDB"
                    if verbose:
                        print(f"   📄 Using seqDB from current taxonomy job")
                else:
                    # Fallback to queryDB in taxonomy directory
                    seqdb_path = f"~/{taxonomy_dir}/queryDB"
                    if verbose:
                        print(f"   📄 Using default queryDB from current taxonomy job")
                        
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Could not parse config, trying to detect database type")
                    print(f"       Error: {e}")
                
                # Try to detect what database exists
                success_seq, check_seqdb = execute_remote_command(sshProc, f"test -f ~/{taxonomy_dir}/seqDB && echo 'seqdb_exists'", timeout=5)
                success_query, check_querydb = execute_remote_command(sshProc, f"test -f ~/{taxonomy_dir}/queryDB && echo 'querydb_exists'", timeout=5)
                
                if "seqdb_exists" in check_seqdb:
                    seqdb_path = f"~/{taxonomy_dir}/seqDB"
                elif "querydb_exists" in check_querydb:
                    seqdb_path = f"~/{taxonomy_dir}/queryDB"
                else:
                    seqdb_path = f"~/{taxonomy_dir}/queryDB"  # Default fallback
                    
        else:
            # Try to detect what database exists
            success_seq, check_seqdb = execute_remote_command(sshProc, f"test -f ~/{taxonomy_dir}/seqDB && echo 'seqdb_exists'", timeout=5)
            if "seqdb_exists" in check_seqdb:
                seqdb_path = f"~/{taxonomy_dir}/seqDB"
            else:
                seqdb_path = f"~/{taxonomy_dir}/queryDB"
        
        # Convert taxonomy results to TSV
        if verbose:
            print("   📄 Converting taxonomy results to TSV format...")
        createtsv_cmd = f"cd ~/{taxonomy_dir} && {mmseqs_path} createtsv {seqdb_path} taxonomyResult taxonomy_results.tsv"
        success, createtsv_result = execute_remote_command(sshProc, createtsv_cmd, timeout=120, error_reporting=verbose)
        
        if not success:
            print(f"❌ Error converting taxonomy results: {createtsv_result}")
            return None
        
        print("   ✅ Taxonomy results converted to TSV format")
        
        # Download results and config files
        remote_results_file = f"~/{taxonomy_dir}/taxonomy_results.tsv"
        remote_config = f"~/{taxonomy_dir}/config.json"
        local_result = f"{local_project_dir}/taxonomy_results.tsv"
        local_config = f"{local_project_dir}/taxonomy_config.json"
        
        try:
            # Download TSV results file
            if verbose:
                print(f"   📥 Downloading taxonomy TSV from: {remote_results_file}")
            scp_from_remote(remote_results_file, local_result, "shr-zion.stanford.edu", USER, PASSWORD)
            print(f"✅ Taxonomy results saved to: {local_result}")
            
            # Download config file
            if verbose:
                print(f"   📥 Downloading config from: {remote_config}")
            scp_from_remote(remote_config, local_config, "shr-zion.stanford.edu", USER, PASSWORD)
            if verbose:
                print(f"✅ Config saved to: {local_config}")
            
        except Exception as e:
            print(f"❌ Failed to retrieve taxonomy files: {e}")
            if verbose:
                print(f"   You can manually retrieve results from: {remote_results_file}")
                print(f"   You can manually retrieve config from: {remote_config}")
            return None
        
        if verbose:
            print(f"🧹 Remote files remain in: {project_dir} (for debugging)")
        
        # Update job status to completed in manifest
        _update_job_status(sshProc, job_id, "taxonomy", "completed")
        
        return {
            "job_type": "taxonomy",
            "results": local_result,
            "config": local_config,
            "job_id": job_id,
            "remote_path": remote_results_file
        }
    
    else:
        # Handle alignment job results
        print("📥 Downloading aligned FASTA file...")
        
        # Generate local output filename in project directory
        local_aligned = f"{local_project_dir}/aligned.fasta"
        remote_aligned = f"~/{align_dir}/aligned.fasta"
        
        try:
            scp_from_remote(remote_aligned, local_aligned, "shr-zion.stanford.edu", USER, PASSWORD)
            print(f"✅ Aligned FASTA saved to: {local_aligned}")
            
            # Show file statistics
            if verbose:
                success, stats_result = execute_remote_command(sshProc, f"grep -c '^>' {remote_aligned} && wc -c {remote_aligned}", timeout=10)
                if success:
                    stats_clean = _clean_terminal_output(stats_result)
                    lines = stats_clean.split('\n')
                    if len(lines) >= 2:
                        seq_count = lines[0].strip()
                        file_size = lines[1].strip().split()[0]
                        print(f"   📊 Sequences: {seq_count}, File size: {file_size} bytes")
            
        except Exception as e:
            print(f"❌ Failed to retrieve aligned file: {e}")
            if verbose:
                print("   You can manually retrieve the file from:", remote_aligned)
            return None
        
        if verbose:
            print(f"🧹 Remote files remain in: {project_dir} (for debugging)")
        
        # Update job status to completed in manifest
        _update_job_status(sshProc, job_id, "align", "completed")
        
        return {
            "job_type": "align",
            "aligned_fasta": local_aligned,
            "job_id": job_id,
            "remote_path": remote_aligned
        }


def status(job_id=None, sshProc=None, verbose=False):
    """
    Check the status of beak jobs using the job manifest system.
    
    Args:
        job_id (str, optional): Specific job ID to check. If None, shows all jobs.
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed debugging output.
    Returns:
        dict: Status information for the job(s).
    """
    if sshProc is None:
        sshProc = sopen()
    
    if verbose:
        print("🔍 Scanning and updating job manifest...")
    
    # Scan directories and update manifest with current status
    manifest = _scan_and_update_jobs(sshProc, verbose)
    
    if job_id:
        # Check specific job
        if job_id not in manifest["jobs"]:
            print(f"❌ Job '{job_id}' not found")
            return {"job_id": job_id, "status": "not_found"}
        
        job_info = manifest["jobs"][job_id]
        operations = job_info["operations"]
        
        if not operations:
            print(f"❌ Job '{job_id}' has no operations")
            return {"job_id": job_id, "status": "unknown"}
        
        # Display project header with all operations
        print(f"📋 Project: {job_id}")
        
        # Show all operations with their status
        operation_order = ["search", "align", "tree", "taxonomy"]  # Preferred display order
        sorted_operations = []
        
        # Add operations in preferred order
        for op_type in operation_order:
            if op_type in operations:
                sorted_operations.append((op_type, operations[op_type]))
        
        # Add any other operations not in the standard list
        for op_type, op_data in operations.items():
            if op_type not in operation_order:
                sorted_operations.append((op_type, op_data))
        
        # Display each operation status
        for op_type, op_data in sorted_operations:
            status_str = op_data["status"]
            
            # Format status with appropriate emoji and styling
            if status_str == "completed":
                status_display = "Complete"
                emoji = "✅"
            elif status_str == "running":
                status_display = "In progress"
                emoji = "🔄"
            elif status_str == "failed":
                status_display = "Failed"
                emoji = "❌"
            else:
                status_display = status_str.title()
                emoji = "❓"
            
            print(f"{emoji} {op_type.title()}: {status_display}")
            
            # Show additional info for completed operations
            if status_str == "completed" and not verbose:
                if op_type == "search":
                    print(f"   Use retrieve_results('{job_id}') to download results")
                elif op_type == "align":
                    print(f"   Alignment file: ~/beak_tmp/{job_id}/align/aligned.fasta")
                elif op_type == "tree":
                    print(f"   Tree file: ~/beak_tmp/{job_id}/tree/")
                elif op_type == "taxonomy":
                    print(f"   Use retrieve_results('{job_id}') to download taxonomy results")
            
            # Show recent log output for running operations in verbose mode
            if status_str == "running" and verbose:
                log_path = f"~/beak_tmp/{job_id}/{op_type}/{op_type}_output.log" if op_type in ["search", "taxonomy"] else f"~/beak_tmp/{job_id}/{op_type}/job.log"
                success, log_content = execute_remote_command(sshProc, f"tail -3 {log_path} 2>/dev/null || echo 'no_log'", timeout=10)
                if success and "no_log" not in log_content:
                    log_clean = log_content.strip()
                    if log_clean:
                        print(f"   Recent: {log_clean.split(chr(10))[-1] if chr(10) in log_clean else log_clean}")
        
        # Determine overall project status for return value
        if all(op["status"] == "completed" for op in operations.values()):
            overall_status = "completed"
        elif any(op["status"] == "running" for op in operations.values()):
            overall_status = "running" 
        elif any(op["status"] == "failed" for op in operations.values()):
            overall_status = "failed"
        else:
            overall_status = "unknown"
        
        return {
            "job_id": job_id,
            "status": overall_status,
            "is_running": overall_status == "running",
            "is_complete": overall_status == "completed",
            "operations": {op_type: op_data["status"] for op_type, op_data in operations.items()},
            "operation_details": operations,
            "created_by": job_info.get("created_by"),
            "last_updated": job_info.get("last_updated")
        }
    
    else:
        # Show all jobs
        jobs = list(manifest["jobs"].values())
        
        if not jobs:
            print("📋 No beak jobs found")
            return {"jobs": []}
        
        # Sort jobs by recency (most recent first)
        jobs_sorted = sorted(jobs, key=lambda x: x.get("last_updated", ""), reverse=True)
        
        print(f"📋 Found {len(jobs_sorted)} project(s) (sorted by recency)")
        print()
        
        for job_info in jobs_sorted:
            job_id = job_info["job_id"]
            operations = job_info["operations"]
            
            if not operations:
                continue
            
            # Show project header
            print(f"📋 Project: {job_id}")
            
            # Show operations in preferred order
            operation_order = ["search", "align", "tree", "taxonomy"]
            sorted_operations = []
            
            # Add operations in preferred order  
            for op_type in operation_order:
                if op_type in operations:
                    sorted_operations.append((op_type, operations[op_type]))
            
            # Add any other operations
            for op_type, op_data in operations.items():
                if op_type not in operation_order:
                    sorted_operations.append((op_type, op_data))
            
            # Show each operation status with emojis
            for op_type, op_data in sorted_operations:
                status_str = op_data["status"]
                
                # Format status with appropriate emoji
                if status_str == "completed":
                    status_display = "Complete"
                    emoji = "✅"
                elif status_str == "running":
                    status_display = "In progress"
                    emoji = "🔄"
                elif status_str == "failed":
                    status_display = "Failed"
                    emoji = "❌"
                else:
                    status_display = status_str.title()
                    emoji = "❓"
                
                print(f"{emoji} {op_type.title()}: {status_display}")
            
            print()  # Add blank line between projects
        
        return {"jobs": jobs_sorted}


def align(input_fasta, output_fasta=None, job_id=None, sshProc=None, verbose=False, user_id=None, debug=False):
    """
    Perform multiple protein sequence alignment using Clustal Omega on the remote server.
    
    Args:
        input_fasta (str): Path to local protein FASTA file to align, protein FASTA content as string, or existing project ID.
        output_fasta (str, optional): Local output file path. If None, generates automatic name.
        job_id (str, optional): Custom job ID or existing job ID to reuse. If None, creates new job.
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed output.
        user_id (str, optional): Custom job identifier (deprecated, use job_id instead).
        debug (bool): If True, show all remote commands and responses.
    Returns:
        dict: {"aligned_fasta": "local_aligned_file.fasta", "job_id": "job_id"} or None on failure.
    """
    import os
    import time
    import uuid
    
    if sshProc is None:
        sshProc = sopen()
    
    central_tmp_dir = "beak_tmp"
    
    # Debug-aware command execution
    def debug_execute(cmd, timeout=30):
        if debug:
            print(f"🔧 DEBUG CMD: {cmd}")
        success, result = execute_remote_command(sshProc, cmd, timeout=timeout, error_reporting=debug)
        if debug:
            print(f"🔧 DEBUG RESULT: success={success}, output='{result}'")
        return success, result
    
    # Handle project_id parameter - could be custom ID or existing project directory
    # Prefer job_id parameter over user_id for backward compatibility
    custom_id = job_id or user_id
    
    # Check if input_fasta is actually a project ID (starts with 'beak_')
    if input_fasta.startswith('beak_') and not os.path.isfile(input_fasta):
        project_id = input_fasta
        if verbose or debug:
            print(f"🔄 Using existing project from input parameter: {project_id}")
    # If job_id is provided and starts with 'beak_', treat as existing project directory
    elif job_id and job_id.startswith('beak_'):
        project_id = job_id
        if verbose or debug:
            print(f"🔄 Reusing existing project from job_id: {project_id}")
    elif os.path.isfile(input_fasta) and ("search_results.fasta" in input_fasta or "mmseqs_search_results_" in input_fasta):
        # Try to extract project_id from search results filename patterns
        filename = os.path.basename(input_fasta)
        
        # Check for new format: beak_results/beak_project_id/search_results.fasta
        if "search_results.fasta" in filename:
            parent_dir = os.path.dirname(input_fasta)
            potential_project_id = os.path.basename(parent_dir)
            if potential_project_id.startswith("beak_"):
                project_id = potential_project_id
                if verbose:
                    print(f"🔄 Extracted project_id from path: {project_id}")
            else:
                # Create new project_id
                project_id = _generate_project_id(custom_id)
        # Check for old format: mmseqs_search_results_beak_project_id.fasta
        elif filename.startswith("mmseqs_search_results_") and filename.endswith(".fasta"):
            extracted_project_id = filename[22:-6]  # Remove prefix and suffix
            if extracted_project_id.startswith("beak_"):
                project_id = extracted_project_id
                if verbose:
                    print(f"🔄 Extracted project_id from filename: {project_id}")
            else:
                # Create new project_id
                project_id = _generate_project_id(custom_id)
        else:
            # Create new project_id
            project_id = _generate_project_id(custom_id)
    else:
        # Create new project_id
        project_id = _generate_project_id(custom_id)
    
    # Get project structure with align subdirectory
    structure = _get_project_structure(project_id, "align")
    project_dir = structure["project_dir"]
    align_dir = structure["operation_dir"]
    
    print(f"🧬 Starting Clustal Omega alignment...")
    if verbose:
        print(f"📁 Project directory: {project_id}")
        print(f"📂 Align operation directory: {align_dir}")
    
    # Determine if input is a file path, FASTA content, or existing project ID
    remote_input_fasta = f"~/{align_dir}/input.fasta"
    
    # Check if input_fasta is actually a project ID and if it has existing search results
    if input_fasta.startswith('beak_') and input_fasta == project_id:
        # This is a project ID - check if it has search results to align
        search_dir = f"{central_tmp_dir}/{project_id}/search"
        search_results_path = f"~/{search_dir}/resultDB"
        
        if verbose:
            print(f"🔍 Checking for search results in project: {project_id}")
            print(f"🔍 Looking for resultDB at: {search_results_path}")
        
        # Check if search results exist (look for resultDB file)
        # Try a simpler approach that might avoid banner pollution
        success, result = debug_execute(f"[ -f {search_results_path} ] && echo exists || echo missing", timeout=10)
        result_clean = _clean_terminal_output(result)
        
        # If that failed, try the alternative syntax
        if not success or (result_clean not in ['exists', 'missing']):
            if verbose:
                print("🔄 Retrying with alternative test command syntax...")
            success, result = debug_execute(f"test -f {search_results_path} && echo 'exists' || echo 'missing'", timeout=10)
            result_clean = _clean_terminal_output(result)
        
        if debug:
            print(f"🔧 DEBUG: Search results check - success: {success}")
            print(f"🔧 DEBUG: Raw result length: {len(result) if result else 0} chars")
            print(f"🔧 DEBUG: Raw result (first 200 chars): '{result[:200] if result else 'None'}...'")
            print(f"🔧 DEBUG: Cleaned result: '{result_clean}'")
        
        if success and "exists" in result_clean:
            if verbose:
                print(f"✅ Found search results, converting to FASTA for alignment...")
            
            # Create align directory
            if debug:
                print(f"🔧 DEBUG CMD: mkdir -p ~/{align_dir}")
            ssend(f"mkdir -p ~/{align_dir}", sshProc, timeout=10)
            
            if verbose:
                print(f"📁 Created align directory: ~/{align_dir}")
            
            # Verify align directory was created
            success, dir_check = debug_execute(f"test -d ~/{align_dir} && echo 'exists' || echo 'missing'", timeout=10)
            if debug:
                print(f"🔧 DEBUG: Align directory check - success: {success}, result: '{dir_check}'")
            
            # Check if search results FASTA already exists
            search_dir = f"{central_tmp_dir}/{project_id}/search"
            search_fasta_path = f"~/{search_dir}/search_results.fasta"
            
            if verbose:
                print(f"🔍 Checking for existing search results FASTA...")
            
            success, fasta_check = debug_execute(f"test -f {search_fasta_path} && echo 'exists' || echo 'missing'", timeout=10)
            fasta_check_clean = _clean_terminal_output(fasta_check)
            
            if success and "exists" in fasta_check_clean:
                # Use existing FASTA file
                if verbose:
                    print(f"✅ Found existing search results FASTA, copying for alignment...")
                
                copy_cmd = f"cp {search_fasta_path} {remote_input_fasta}"
                success, copy_result = debug_execute(copy_cmd, timeout=30)
                if not success:
                    print(f"❌ Failed to copy search results FASTA")
                    return None
            else:
                # Need to extract sequences from search results database
                if verbose:
                    print(f"🔄 Extracting FASTA sequences from search results database...")
                
                # Get the target database path from config
                search_config_path = f"~/{search_dir}/config.json"
                success, config_content = debug_execute(f"cat {search_config_path} 2>/dev/null", timeout=10)
                
                if success and config_content:
                    try:
                        import json
                        config_clean = _clean_terminal_output(config_content)
                        config = json.loads(config_clean)
                        target_db = config.get("target_db_path", "/data/databases/uniref90/uniref90")
                    except:
                        target_db = "/data/databases/uniref90/uniref90"  # Default
                else:
                    target_db = "/data/databases/uniref90/uniref90"  # Default
                
                # Use mmseqs to extract target sequences from search results
                extract_cmd = f"cd ~/{search_dir} && mmseqs result2flat {target_db} {target_db} resultDB {remote_input_fasta} --use-fasta-header"
                
                if verbose:
                    print(f"🔄 Extracting sequences: {extract_cmd}")
                
                success, extract_result = debug_execute(extract_cmd, timeout=120)
                if not success:
                    print(f"❌ Failed to extract sequences from search results")
                    if verbose:
                        print(f"   Error: {extract_result}")
                    return None
                
            input_source = f"Project search results: {project_id}"
        else:
            print(f"❌ No search results found in project {project_id}")
            if debug:
                # Let's check what's actually in the search directory
                print(f"🔧 DEBUG: Investigating search directory structure...")
                success, dir_list = debug_execute(f"ls -la ~/{search_dir}/ 2>/dev/null || echo 'directory not found'", timeout=10)
                if success:
                    print(f"🔧 DEBUG: Search directory contents:\n{dir_list}")
                
                # Also check if the directory exists at all
                success, dir_exists = debug_execute(f"test -d ~/{search_dir} && echo 'directory exists' || echo 'directory missing'", timeout=10)
                print(f"🔧 DEBUG: Search directory exists check: {dir_exists}")
                
                # Check alternative path structures
                alt_search_dir = f"beak_tmp/{project_id}/search"
                success, alt_check = debug_execute(f"test -d ~/{alt_search_dir} && echo 'found alternative path' || echo 'not found'", timeout=10)
                print(f"🔧 DEBUG: Alternative search path ~/{alt_search_dir}: {alt_check}")
            
            print(f"   Expected resultDB at: {search_results_path}")
            print(f"   Run a search first or check if the project directory exists.")
            return None
            
    elif os.path.isfile(input_fasta):
        # Upload FASTA file to remote
        if verbose:
            print("📤 Uploading FASTA file to remote server...")
        
        # Create align directory
        if debug:
            print(f"🔧 DEBUG CMD: mkdir -p ~/{align_dir}")
        ssend(f"mkdir -p ~/{align_dir}", sshProc, timeout=10)
        
        if verbose:
            print(f"📁 Created align directory: ~/{align_dir}")
        scp_to_remote(input_fasta, remote_input_fasta, "shr-zion.stanford.edu", USER, PASSWORD)
        input_source = f"File: {input_fasta}"
    else:
        # Treat input as FASTA content string
        if verbose:
            print("✍️  Writing FASTA content to remote file...")
        
        # Use a more robust method to write the FASTA file
        import tempfile
        
        # Write FASTA content to a local temp file first
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
            tmp_file.write(input_fasta)
            local_temp_fasta = tmp_file.name
        
        try:
            # Create align directory
            if debug:
                print(f"🔧 DEBUG CMD: mkdir -p ~/{align_dir}")
            ssend(f"mkdir -p ~/{align_dir}", sshProc, timeout=10)
            
            if verbose:
                print(f"📁 Created align directory: ~/{align_dir}")
            scp_to_remote(local_temp_fasta, remote_input_fasta, "shr-zion.stanford.edu", USER, PASSWORD)
            if verbose:
                print(f"   ✅ FASTA content uploaded to: {remote_input_fasta}")
        finally:
            # Clean up local temp file
            os.unlink(local_temp_fasta)
        
        input_source = "FASTA content string"
    
    # Create the asynchronous alignment script
    print("🧹 Preparing alignment job for background execution...")
    
    # Create alignment script that runs in background
    alignment_script = f"""#!/bin/bash
cd ~/{align_dir}

# Redirect all output to log file for monitoring
exec > job.log 2>&1

echo "$(date): Starting alignment job {project_id}"
echo "STEP1: Cleaning sequences"
awk '/^>/{{if(s){{print h"\\n"s}};h=$0;s="";next}}{{s=s $0 "\\n"}}END{{if(s)print h"\\n"s}}' input.fasta > input_clean.fasta

echo "STEP2: Input sequences: $(grep -c '^>' input.fasta)"
echo "STEP3: Cleaned sequences: $(grep -c '^>' input_clean.fasta)"

echo "STEP4: Checking clustalo installation"
(which clustalo >/dev/null 2>&1 || conda install -c bioconda clustalo -y >/dev/null 2>&1)

echo "STEP5: Running Clustal Omega alignment"
echo "$(date): Starting clustalo process"
clustalo -i input_clean.fasta -o aligned.fasta --force --verbose

if [ $? -eq 0 ]; then
    echo "STEP6: Alignment completed successfully"
    echo "STEP7: Output sequences: $(grep -c '^>' aligned.fasta)"
    echo "STEP8: Output file size: $(wc -c < aligned.fasta) bytes"
    echo "$(date): SUCCESS_COMPLETE"
    echo "completed" > .status
else
    echo "$(date): ALIGNMENT_FAILED"
    echo "failed" > .status
fi

# Clean up PID file
rm -f .pid
"""
    
    # Write script to remote
    script_path = f"~/{align_dir}/run_alignment.sh"
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as tmp_file:
        tmp_file.write(alignment_script)
        local_temp_script = tmp_file.name
    
    try:
        scp_to_remote(local_temp_script, script_path, "shr-zion.stanford.edu", USER, PASSWORD)
        if verbose:
            print(f"   ✅ Alignment script uploaded to: {script_path}")
    except Exception as e:
        print(f"❌ Failed to upload alignment script: {e}")
        return None
    finally:
        os.unlink(local_temp_script)
    
    # Make script executable and start it in background
    make_executable_cmd = f"chmod +x {script_path}"
    success, result = debug_execute(make_executable_cmd, timeout=10)
    if not success:
        print(f"❌ Failed to make script executable")
        return None
    
    # Start background job and capture PID
    background_cmd = f"cd ~/{align_dir} && nohup bash run_alignment.sh > /dev/null 2>&1 & echo $!"
    success, pid_result = debug_execute(background_cmd, timeout=10)
    
    if not success:
        print(f"❌ Failed to start background alignment job")
        return None
    
    # Clean and validate PID
    pid_clean = _clean_terminal_output(pid_result)
    if not pid_clean or not pid_clean.isdigit():
        print(f"❌ Failed to get valid process ID from background job")
        if debug:
            print(f"🔧 DEBUG: Raw PID result: '{pid_result}'")
            print(f"🔧 DEBUG: Cleaned PID result: '{pid_clean}'")
        return None
    
    # Store PID for monitoring
    pid_cmd = f"echo {pid_clean} > ~/{align_dir}/.pid"
    success, _ = debug_execute(pid_cmd, timeout=10)
    
    # Wait a moment and verify job started successfully
    import time
    time.sleep(2)
    
    # Check if process is still running
    verify_cmd = f"kill -0 {pid_clean} 2>/dev/null && echo 'running' || echo 'not_running'"
    success, verify_result = debug_execute(verify_cmd, timeout=10)
    verify_result_clean = _clean_terminal_output(verify_result)
    
    if "running" not in verify_result_clean:
        print(f"❌ Alignment process failed to start or died immediately")
        # Check for immediate errors in log
        success, log_result = debug_execute(f"tail -5 ~/{align_dir}/job.log 2>/dev/null || echo 'no_log'", timeout=10)
        if success and "no_log" not in log_result:
            print(f"   Recent log output: {_clean_terminal_output(log_result)}")
        return None
    
    print(f"✅ Alignment job started successfully!")
    print(f"   🆔 Job ID: {project_id}")
    print(f"   🔢 Process ID: {pid_clean}")
    print(f"   📁 Remote directory: {align_dir}")
    print(f"   📋 Monitor progress with: status('{project_id}')")
    
    # Create config file for the align operation
    from datetime import datetime
    config_data = {
        "project_id": project_id,
        "operation_type": "align",
        "timestamp": datetime.now().isoformat(),
        "input_source": input_source,
        "project_directory": project_dir,
        "align_directory": align_dir,
        "clustalo_command": "clustalo -i input_clean.fasta -o aligned.fasta --force --verbose",
        "process_id": pid_clean,
        "status": "running"
    }
    
    # Write config file to remote
    remote_config = structure["config_file"]
    config_json = json.dumps(config_data, indent=2)
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_file.write(config_json)
        local_temp_config = tmp_file.name
    
    try:
        scp_to_remote(local_temp_config, remote_config, "shr-zion.stanford.edu", USER, PASSWORD)
        if verbose:
            print(f"   ✅ Alignment config saved to: {remote_config}")
            print(f"      (Operation-specific config in align/ subdirectory)")
    except Exception as e:
        if verbose:
            print(f"   ⚠️  Warning: Failed to save config: {e}")
    finally:
        os.unlink(local_temp_config)
    
    # Register job as running in manifest
    _register_job(sshProc, project_id, "align", "running")
    
    return {
        "job_id": project_id,
        "project_id": project_id,
        "status": "running",
        "process_id": pid_clean,
        "input_source": input_source,
        "remote_dir": project_dir,
        "align_dir": align_dir,
        "log_file": f"{align_dir}/job.log",
        "expected_output": f"{align_dir}/aligned.fasta"
    }


def compute_tree(input_source, tree_method="ML", job_id=None, sshProc=None, verbose=False, user_id=None, debug=False):
    """
    Compute a maximum likelihood phylogenetic tree using IQ-TREE2 from a FASTA alignment.
    
    Args:
        input_source (str): Path to local FASTA alignment file, FASTA content string, or existing project ID.
        tree_method (str): Tree method to use ("ML" for maximum likelihood). Default: "ML".
        job_id (str, optional): Custom job identifier for tracking.
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed progress output.
        user_id (str, optional): Custom user identifier for job naming.
        debug (bool): If True, keep temporary files and show debug output.
        
    Returns:
        dict: {"tree_file": "local_tree_file.treefile", "job_id": "job_id", "log_file": "iqtree_log.log"} or None on failure.
    """
    import os
    import time
    import json
    from datetime import datetime
    
    if sshProc is None:
        sshProc = sopen()
    
    # Generate project ID
    if job_id:
        if job_id.startswith("beak_"):
            project_id = job_id
        else:
            project_id = f"beak_{job_id}"
    else:
        if user_id:
            project_id = f"beak_{user_id}_{int(time.time())}"
        else:
            project_id = _generate_unique_id()
    
    # Get project structure with tree subdirectory
    structure = _get_project_structure(project_id, "tree")
    project_dir = structure["project_dir"]
    tree_dir = structure["operation_dir"]
    
    print(f"🌳 Starting IQ-TREE2 phylogenetic analysis...")
    
    if verbose:
        print(f"📂 Tree operation directory: {tree_dir}")
        print(f"🆔 Project ID: {project_id}")
    
    remote_input_fasta = f"~/{tree_dir}/alignment.fasta"
    
    # Handle different input types
    if input_source.startswith("beak_") and len(input_source.split("_")) >= 2:
        # This is a project ID - check if it has alignment results
        parent_project_dir = f"beak_tmp/{input_source}"
        success, check_aligned = execute_remote_command(sshProc, f"test -f ~/{parent_project_dir}/align/aligned.fasta && echo 'exists' || echo 'missing'", timeout=10)
        check_aligned_clean = _clean_terminal_output(check_aligned)
        
        if "missing" in check_aligned_clean:
            print(f"❌ No alignment file found for project '{input_source}'")
            print(f"   Expected alignment file at: ~/{parent_project_dir}/align/aligned.fasta")
            print("   Please run align() first on this project or provide a FASTA file")
            return None
        
        if "exists" in check_aligned_clean:
            if verbose:
                print(f"✅ Found alignment file, using for tree computation...")
            
        # Create tree directory
        if debug:
            print(f"🔧 DEBUG CMD: mkdir -p ~/{tree_dir}")
        ssend(f"mkdir -p ~/{tree_dir}", sshProc, timeout=10)
        
        # Copy alignment file to tree directory
        copy_cmd = f"cp ~/{parent_project_dir}/align/aligned.fasta {remote_input_fasta}"
        if debug:
            print(f"🔧 DEBUG CMD: {copy_cmd}")
        ssend(copy_cmd, sshProc, timeout=30)
        
    elif os.path.isfile(input_source):
        # Local FASTA file
        if verbose:
            print(f"📁 Uploading local alignment file: {input_source}")
        
        # Create tree directory
        if debug:
            print(f"🔧 DEBUG CMD: mkdir -p ~/{tree_dir}")
        ssend(f"mkdir -p ~/{tree_dir}", sshProc, timeout=10)
        
        # Upload the alignment file
        try:
            scp_to_remote(input_source, remote_input_fasta, "shr-zion.stanford.edu", USER, PASSWORD)
            print(f"✅ Alignment uploaded successfully")
        except Exception as e:
            print(f"❌ Failed to upload alignment file: {e}")
            return None
            
    else:
        # Assume it's FASTA content string
        if verbose:
            print("📝 Processing FASTA content string...")
        
        # Create tree directory
        if debug:
            print(f"🔧 DEBUG CMD: mkdir -p ~/{tree_dir}")
        ssend(f"mkdir -p ~/{tree_dir}", sshProc, timeout=10)
        
        # Write FASTA content to remote file
        fasta_lines = input_source.strip().split('\n')
        fasta_content = '\n'.join(fasta_lines)
        
        write_cmd = f"cat > {remote_input_fasta} << 'EOFFASTA'\n{fasta_content}\nEOFFASTA"
        if debug:
            print(f"🔧 DEBUG CMD: {write_cmd[:100]}...")
        ssend(write_cmd, sshProc, timeout=30)
    
    # Validate alignment file has sequences
    print("🔍 Validating alignment file...")
    success, seq_count_result = execute_remote_command(sshProc, f"grep -c '^>' {remote_input_fasta} 2>/dev/null || echo '0'", timeout=10)
    
    if success:
        seq_count = int(_clean_terminal_output(seq_count_result))
        if seq_count < 3:
            print(f"❌ Insufficient sequences for tree computation (found {seq_count}, need ≥3)")
            print("   Tree reconstruction requires at least 3 sequences")
            return None
        print(f"✅ Found {seq_count} sequences - proceeding with tree computation")
    else:
        print("⚠️ Could not validate sequence count - proceeding anyway")
    
    # Run IQ-TREE2 for maximum likelihood tree
    print("🌳 Running IQ-TREE2 analysis...")
    
    iqtree_cmd = f"""cd ~/{tree_dir} && \\
echo "STEP1: Starting IQ-TREE2 analysis" && \\
echo "Input sequences: $(grep -c '^>' alignment.fasta)" && \\
echo "STEP2: Running model selection and ML tree inference" && \\
iqtree2 -s alignment.fasta -m MFP -bb 1000 -alrt 1000 -nt AUTO --quiet 2>&1 | grep -E "(Analysis|Model|Tree|Log-likelihood|Time)" | head -20 && \\
echo "STEP3: Tree computation completed" && \\
echo "Output files:" && \\
ls -la *.treefile *.log *.iqtree 2>/dev/null || echo "No output files found" && \\
echo "STEP4: Final tree file size: $(wc -c < alignment.fasta.treefile 2>/dev/null || echo '0') bytes" """
    
    if debug:
        print(f"🔧 DEBUG CMD: {iqtree_cmd}")
    
    print("⏳ Running tree inference (this may take several minutes)...")
    success, result = execute_remote_command(sshProc, iqtree_cmd, timeout=1800)  # 30 minute timeout
    
    if not success:
        print(f"❌ Tree computation failed")
        if debug and result:
            print(f"   Error output: {result}")
        return None
    
    if verbose:
        print("🔍 IQ-TREE2 output:")
        print(result)
    
    # Check if tree file was created
    remote_tree_file = f"~/{tree_dir}/alignment.fasta.treefile"
    success, tree_check = execute_remote_command(sshProc, f"test -f {remote_tree_file} && echo 'exists' || echo 'missing'", timeout=10)
    
    if "missing" in _clean_terminal_output(tree_check):
        print("❌ Tree file was not created - IQ-TREE2 may have failed")
        # Try to get error log
        success, log_check = execute_remote_command(sshProc, f"test -f ~/{tree_dir}/alignment.fasta.log && cat ~/{tree_dir}/alignment.fasta.log | tail -20 || echo 'No log found'", timeout=10)
        if success and log_check:
            print("📋 IQ-TREE2 log (last 20 lines):")
            print(log_check)
        return None
    
    # Download tree file and log
    local_project_dir = _get_local_project_dir(project_id)
    os.makedirs(local_project_dir, exist_ok=True)
    
    output_tree = f"{local_project_dir}/tree.treefile"
    output_log = f"{local_project_dir}/iqtree.log"
    
    print("📥 Downloading tree files...")
    
    try:
        # Download tree file
        scp_from_remote(remote_tree_file, output_tree, "shr-zion.stanford.edu", USER, PASSWORD)
        print(f"✅ Tree file saved to: {output_tree}")
        
        # Download log file
        remote_log_file = f"~/{tree_dir}/alignment.fasta.log"
        try:
            scp_from_remote(remote_log_file, output_log, "shr-zion.stanford.edu", USER, PASSWORD)
            if verbose:
                print(f"✅ Log file saved to: {output_log}")
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not download log file: {e}")
        
        # Show tree statistics
        if verbose:
            print("📊 Getting tree statistics...")
            success, stats_result = execute_remote_command(sshProc, f"wc -c {remote_tree_file} && grep -o ')' {remote_tree_file} | wc -l", timeout=10)
            if success and stats_result:
                lines = stats_result.strip().split('\n')
                if len(lines) >= 2:
                    file_size = lines[0].split()[0]
                    node_count = lines[1].strip()
                    print(f"   📈 Tree file size: {file_size} bytes")
                    print(f"   🌿 Internal nodes: {node_count}")
        
    except Exception as e:
        print(f"❌ Failed to retrieve tree file: {e}")
        if debug:
            print("   You can manually retrieve the file from:", remote_tree_file)
        return None
    
    # Create config file for the tree operation
    config = {
        "job_id": project_id,
        "operation_type": "tree",
        "timestamp": datetime.now().isoformat(),
        "tree_method": tree_method,
        "input_source": str(input_source),
        "tree_directory": tree_dir,
        "iqtree_command": "iqtree2 -s alignment.fasta -m MFP -bb 1000 -alrt 1000 -nt AUTO"
    }
    
    config_file = f"{local_project_dir}/tree_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    if verbose:
        print(f"✅ Configuration saved to: {config_file}")
    
    # Clean up remote files if not in debug mode
    if not debug:
        cleanup_cmd = f"rm -rf ~/{tree_dir}"
        ssend(cleanup_cmd, sshProc, timeout=30)
        if verbose:
            print("🧹 Remote temporary files cleaned up")
    else:
        print(f"🧹 Remote files remain in: {tree_dir} (for debugging)")
    
    # Register job completion
    _register_job(sshProc, project_id, "tree", "completed")
    
    return {
        "tree_file": output_tree,
        "log_file": output_log,
        "job_id": project_id,
        "config": config_file,
        "tree_dir": tree_dir
    }


def taxonomy(input_source, sshProc=None, verbose=False, user_id=None):
    """
    Perform taxonomic classification using mmseqs taxonomy on protein sequences.
    
    Args:
        input_source (str): Either a job_id from previous search(), or path to FASTA file, or FASTA content string.
        sshProc: An open SSH process (from sopen()).
        verbose (bool): If True, show detailed output.
        user_id (str, optional): Custom job identifier.
    Returns:
        dict: {"job_id": "job_id", "status": "running", "config": "config_file.json"} or None on failure.
    """
    import os
    import time
    import json
    from datetime import datetime
    
    if sshProc is None:
        sshProc = sopen()
    
    central_tmp_dir = "beak_tmp"
    
    # Determine input type and setup
    if input_source.startswith("beak_") and len(input_source.split("_")) >= 2:
        # This looks like a project_id from previous search
        parent_project_id = input_source
        parent_project_dir = f"{central_tmp_dir}/{parent_project_id}"
        parent_search_dir = f"{parent_project_dir}/search"
        
        # Check if the parent project exists and has a queryDB in search subdirectory
        success, check_querydb = execute_remote_command(sshProc, f"test -f ~/{parent_search_dir}/queryDB && echo 'exists' || echo 'missing'", timeout=10)
        check_querydb_clean = _clean_terminal_output(check_querydb)
        
        if "missing" in check_querydb_clean:
            print(f"❌ Parent project '{parent_project_id}' not found or missing queryDB")
            print("   Make sure the search job completed successfully")
            print(f"   Expected queryDB at: ~/{parent_search_dir}/queryDB")
            return None
        
        print(f"🔍 Starting taxonomic classification for project: {parent_project_id}")
        input_type = "project_id"
        query_db_source = f"~/{parent_search_dir}/queryDB"
        
        if verbose:
            print(f"   📁 Using queryDB from: {query_db_source}")
        
    else:
        # Input is either a file path or FASTA content
        if os.path.isfile(input_source):
            print("🔍 Starting taxonomic classification from FASTA file")
            input_type = "file"
            fasta_content = None
        else:
            print("🔍 Starting taxonomic classification from FASTA content")
            input_type = "content"
            fasta_content = input_source
        
        query_db_source = None  # Will be created
    
    # Use the same project if input comes from existing project, otherwise generate new project
    if input_type == "project_id":
        # Reuse the existing project and create taxonomy subdirectory
        project_id = parent_project_id
        structure = _get_project_structure(project_id, "taxonomy")
    else:
        # Generate unique project ID for new taxonomy project
        project_id = _generate_project_id(user_id)
        structure = _get_project_structure(project_id, "taxonomy")
    
    project_dir = structure["project_dir"]
    taxonomy_dir = structure["operation_dir"]
    
    if verbose:
        print(f"📁 Project: {project_id}")
        print(f"📂 Taxonomy operation directory: {taxonomy_dir}")
    
    # Create project directory structure
    ssend(f"mkdir -p ~/{project_dir}", sshProc, timeout=10)
    ssend(f"mkdir -p ~/{taxonomy_dir}", sshProc, timeout=10)
    
    # Setup queryDB if not from existing project
    if input_type != "project_id":
        print("📝 Setting up sequence database...")
        remote_input_fasta = f"~/{taxonomy_dir}/input.fasta"
        
        if input_type == "file":
            # Upload FASTA file
            if verbose:
                print("   📤 Uploading FASTA file to remote server...")
            scp_to_remote(input_source, remote_input_fasta, "shr-zion.stanford.edu", USER, PASSWORD)
        else:
            # Write FASTA content to remote file
            if verbose:
                print("   ✍️  Writing FASTA content to remote file...")
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as tmp_file:
                tmp_file.write(fasta_content)
                local_temp_fasta = tmp_file.name
            
            try:
                scp_to_remote(local_temp_fasta, remote_input_fasta, "shr-zion.stanford.edu", USER, PASSWORD)
            finally:
                os.unlink(local_temp_fasta)
        
        # Create mmseqs database from FASTA
        if verbose:
            print("   🗄️  Creating mmseqs database...")
        
        # Find mmseqs path
        success, mmseqs_check = execute_remote_command(sshProc, "which mmseqs 2>/dev/null || echo 'mmseqs_not_found'", timeout=10)
        mmseqs_check_clean = mmseqs_check.strip() if mmseqs_check else ""
        
        # Clean any markers that might have leaked through
        if "BEAK_END_" in mmseqs_check_clean:
            mmseqs_check_clean = mmseqs_check_clean.split("BEAK_END_")[0].strip()
        
        if not success or not mmseqs_check_clean or "mmseqs_not_found" in mmseqs_check_clean:
            mmseqs_path = "mmseqs"  # Hope it's in PATH
            if verbose:
                print("   ⚠️  mmseqs path not found, using 'mmseqs'")
        else:
            # Take only the first line and ensure it's a valid path
            first_line = mmseqs_check_clean.split('\n')[0].strip()
            if first_line and '/' in first_line and not first_line.startswith('BEAK_'):
                mmseqs_path = first_line
            else:
                mmseqs_path = "mmseqs"
            if verbose:
                print(f"   ✅ Found mmseqs at: {mmseqs_path}")
        
        # Create seqDB for taxonomy (different from queryDB for search)
        create_db_cmd = f"cd ~/{taxonomy_dir} && {mmseqs_path} createdb input.fasta seqDB"
        success, create_result = execute_remote_command(sshProc, create_db_cmd, timeout=60, error_reporting=verbose)
        
        if not success:
            print(f"❌ Failed to create sequence database: {create_result}")
            return None
        
        print("   ✅ Sequence database (seqDB) created")
        query_db_source = f"~/{taxonomy_dir}/seqDB"
    else:
        # Use existing queryDB from parent job
        mmseqs_path = "mmseqs"  # Will find it later
    
    # Create config file
    if verbose:
        print("📄 Creating taxonomy config file...")
    
    config_data = {
        "project_id": project_id,
        "operation_type": "taxonomy",
        "timestamp": datetime.now().isoformat(),
        "input_type": input_type,
        "parent_project_id": input_source if input_type == "project_id" else None,
        "input_source": input_source if input_type != "project_id" else f"Project: {input_source}",
        "database": "UniRef90",
        "database_path": "/srv/protein_sequence_databases/UniRef90",
        "project_directory": project_dir,
        "taxonomy_directory": taxonomy_dir,
        "mmseqs_command": "mmseqs taxonomy",
        "settings": {
            "tax_lineage": 1,
            "sensitivity": 1
        }
    }
    
    config_json = json.dumps(config_data, indent=2)
    remote_config = structure["config_file"]
    
    # Write config file to remote
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        tmp_file.write(config_json)
        local_temp_config = tmp_file.name
    
    try:
        scp_to_remote(local_temp_config, remote_config, "shr-zion.stanford.edu", USER, PASSWORD)
        if verbose:
            print(f"   ✅ Taxonomy config saved to: {remote_config}")
            print(f"      (Operation-specific config in taxonomy/ subdirectory)")
    finally:
        os.unlink(local_temp_config)
    
    # Find mmseqs path if not already found
    if mmseqs_path == "mmseqs":
        success, mmseqs_check = execute_remote_command(sshProc, "which mmseqs 2>/dev/null || echo 'mmseqs_not_found'", timeout=10)
        mmseqs_check_clean = mmseqs_check.strip() if mmseqs_check else ""
        
        # Clean any markers that might have leaked through
        if "BEAK_END_" in mmseqs_check_clean:
            mmseqs_check_clean = mmseqs_check_clean.split("BEAK_END_")[0].strip()
        
        if success and mmseqs_check_clean and "mmseqs_not_found" not in mmseqs_check_clean:
            # Take only the first line and ensure it's a valid path
            first_line = mmseqs_check_clean.split('\n')[0].strip()
            if first_line and '/' in first_line and not first_line.startswith('BEAK_'):
                mmseqs_path = first_line
                if verbose:
                    print(f"   ✅ Found mmseqs at: {mmseqs_path}")
            else:
                print("❌ mmseqs path detection failed. Make sure mmseqs2 is installed on the remote server.")
                return None
        else:
            print("❌ mmseqs not found. Make sure mmseqs2 is installed on the remote server.")
            return None
    
    # Run taxonomic classification
    print(f"🧬 Running mmseqs taxonomic classification...")
    if not verbose:
        print("   ⏳ This may take several minutes...")
    else:
        print("   ⏳ This may take 10-30 minutes for large datasets...")
    
    # Validate queryDB exists and is accessible
    if verbose:
        print(f"   🔍 Validating queryDB at: {query_db_source}")
    success, db_check = execute_remote_command(sshProc, f"test -f {query_db_source} && echo 'db_exists' || echo 'db_missing'", timeout=10)
    db_check_clean = _clean_terminal_output(db_check)
    
    if "db_missing" in db_check_clean:
        print(f"❌ QueryDB not found at: {query_db_source}")
        print("   Make sure the input source is valid and accessible")
        return None
    elif verbose:
        print(f"   ✅ QueryDB validated successfully")
    
    # Setup the taxonomy command
    db_path = "/srv/protein_sequence_databases/UniRef90"
    taxonomy_cmd = (
        f"cd ~/{taxonomy_dir} && {mmseqs_path} taxonomy "
        f"{query_db_source} {db_path} taxonomyResult tmp --tax-lineage 1"
    )
    
    if verbose:
        print(f"   🔧 Command: {taxonomy_cmd}")
    
    # Create a background process with tracking
    background_script = f"""
cd ~/{taxonomy_dir} && \\
echo "TAXONOMY_STARTED" > .running && \\
echo "$$" > .pid && \\
({taxonomy_cmd} > taxonomy_output.log 2>&1; EXIT_CODE=$?; \\
 if [ $EXIT_CODE -eq 0 ]; then echo "TAXONOMY_SUCCESS" > .status; else echo "TAXONOMY_FAILED" > .status; fi; \\
 rm -f .running .pid; exit $EXIT_CODE) &
"""
    
    if verbose:
        print(f"   📝 Background script:")
        print(f"   {background_script.strip()}")
        print(f"   📝 Starting tracked background process...")
    
    # Start the tracked background taxonomy search
    ssend(background_script.strip(), sshProc, timeout=10)
    
    # Wait a moment for the process to start
    time.sleep(3)
    
    # Check if there are any immediate errors
    if verbose:
        print("   🔍 Checking for immediate errors...")
        success, log_check = execute_remote_command(sshProc, f"head -10 ~/{taxonomy_dir}/taxonomy_output.log 2>/dev/null || echo 'no_log_yet'", timeout=5)
        if success and "no_log_yet" not in log_check:
            print(f"   📋 Initial log content: {log_check}")
    
    # Verify the taxonomy search started
    success, process_info = execute_remote_command(sshProc, "ps aux | grep mmseqs | grep taxonomy | grep -v grep", timeout=5)
    if success and process_info.strip():
        print("   ✅ mmseqs taxonomy search is running")
        if verbose:
            print(f"   🔍 Process info: {process_info.strip()}")
    else:
        print("   ⚠️  mmseqs taxonomy process not found in process list")
        # Check if there's an error message in the log
        success, error_check = execute_remote_command(sshProc, f"cat ~/{taxonomy_dir}/taxonomy_output.log 2>/dev/null || echo 'no_log'", timeout=5)
        if success and "no_log" not in error_check:
            print(f"   ❌ Error in log: {error_check}")
    
    # Download config file immediately to local project directory
    local_project_dir = _get_local_project_dir(project_id)
    os.makedirs(local_project_dir, exist_ok=True)
    local_config = f"{local_project_dir}/taxonomy_config.json"
    try:
        scp_from_remote(remote_config, local_config, "shr-zion.stanford.edu", USER, PASSWORD)
        if verbose:
            print(f"✅ Config saved to: {local_config}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to download config: {e}")
        local_config = None
    
    print(f"🚀 Taxonomy job '{project_id}' started on remote server")
    print(f"   Use status('{project_id}') to check progress")
    print(f"   Use retrieve_results('{project_id}') when complete")
    
    # Register job in manifest
    _register_job(sshProc, project_id, "taxonomy", "running")
    
    return {
        "job_id": project_id,
        "project_id": project_id,
        "operation_type": "taxonomy",
        "status": "running",
        "config": local_config,
        "remote_dir": project_dir,
        "taxonomy_dir": taxonomy_dir,
        "parent_project_id": input_source if input_type == "project_id" else None
    }



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


def simple_projects_test(sshProc=None):
    """Simple test function using basic commands to validate remote access"""
    if sshProc is None:
        sshProc = sopen()
    
    print("🧪 Simple Projects Test")
    print("=" * 50)
    
    # Test 1: Use simple ls command that should work
    print("\n1. Testing basic ls command:")
    result = ssend("ls ~/beak_tmp", sshProc, timeout=10)
    print(f"   Raw result: '{result}'")
    # print(f"   Lines: {result.split('\\n') if result else 'None'}")
    
    # Test 2: Use the same command as working functions
    print("\n2. Testing working pattern from _scan_and_update_jobs:")
    result = ssend("ls -1 ~/beak_tmp/ 2>/dev/null | grep '^beak_' || echo 'no_projects'", sshProc, timeout=10)
    print(f"   Raw result: '{result}'")
    cleaned = _clean_terminal_output(result)
    print(f"   Cleaned result: '{cleaned}'")
    
    # Test 3: Manually parse what we get
    if result:
        lines = result.split('\\n')
        beak_dirs = [line.strip() for line in lines if line.strip().startswith('beak_')]
        print(f"   Found beak directories: {beak_dirs}")
    
    return result


def debug_projects(sshProc=None):
    """Debug function to test remote directory access"""
    if sshProc is None:
        sshProc = sopen()
    
    print("🔍 Debug: Testing remote directory access...")
    
    # Test 1: Check if beak_tmp exists
    print("\n1. Testing beak_tmp directory existence:")
    success, result = execute_remote_command(sshProc, "test -d ~/beak_tmp && echo 'EXISTS' || echo 'MISSING'", timeout=10)
    print(f"   Success: {success}")
    print(f"   Result: '{result}'")
    
    # Test 2: List beak_tmp contents with detailed info
    print("\n2. Listing beak_tmp contents (detailed):")
    success, result = execute_remote_command(sshProc, "ls -la ~/beak_tmp/", timeout=10)
    print(f"   Success: {success}")
    print(f"   Result: '{result}'")
    
    # Test 3: List just directory names
    print("\n3. Listing directory names only:")
    success, result = execute_remote_command(sshProc, "ls -1 ~/beak_tmp/", timeout=10)
    print(f"   Success: {success}")
    print(f"   Result: '{result}'")
    
    # Test 4: Check for beak_ prefixed directories
    print("\n4. Looking for beak_ prefixed directories:")
    success, result = execute_remote_command(sshProc, "ls -1d ~/beak_tmp/beak_* 2>/dev/null || echo 'NO_BEAK_DIRS'", timeout=10)
    print(f"   Success: {success}")
    print(f"   Result: '{result}'")
    
    # Test 5: Alternative approach with find
    print("\n5. Using find to locate beak directories:")
    success, result = execute_remote_command(sshProc, "find ~/beak_tmp -maxdepth 1 -type d -name 'beak_*' 2>/dev/null || echo 'FIND_FAILED'", timeout=10)
    print(f"   Success: {success}")
    print(f"   Result: '{result}'")
    
    return None


def projects(sshProc=None, verbose=False):
    """
    List all existing projects in the user's remote directory and their contents.
    
    This function scans the remote beak_tmp directory for projects and displays
    information about search results, alignments, and taxonomy operations within
    each project, including counts of sequences and hits.
    
    Args:
        sshProc: An open SSH process (from sopen()). If None, opens a new connection.
        verbose (bool): If True, show detailed debugging output.
    
    Returns:
        dict: Project information organized by project name.
    """
    if sshProc is None:
        sshProc = sopen()
    
    # Use the same proven approach as _scan_and_update_jobs
    central_tmp_dir = "beak_tmp"
    
    # Use the WORKING command from debug_projects (test #4)
    success, dir_list = execute_remote_command(sshProc, f"ls -1d ~/{central_tmp_dir}/beak_* 2>/dev/null || echo 'no_projects'", timeout=10)
    
    if verbose:
        print(f"🔍 Raw directory listing result: success={success}, output='{dir_list}'")
    
    # Custom cleaning for directory listings (handles multiple lines better than _clean_terminal_output)
    def clean_directory_listing(output):
        if not output:
            return []
        
        # Remove BEAK_END markers
        import re
        cleaned = re.sub(r'BEAK_END_\d+_\d+', '', output)
        
        # Split into lines and find beak directories
        lines = cleaned.split('\n')
        beak_dirs = []
        
        for line in lines:
            line = line.strip()
            # Handle both directory names and full paths
            if ('beak_' in line and not any(bad in line.lower() for bad in ['error', 'not found', 'failed', 'no such'])):
                # Extract just the directory name from full path if needed
                if '/' in line and line.startswith('/'):
                    # Full path: /home/user/beak_tmp/beak_project_name
                    dir_name = line.split('/')[-1]
                    if dir_name.startswith('beak_'):
                        beak_dirs.append(dir_name)
                elif line.startswith('beak_'):
                    # Just directory name: beak_project_name
                    beak_dirs.append(line)
        
        return beak_dirs
    
    project_dirs = clean_directory_listing(dir_list)
    
    if verbose:
        print(f"🔍 Found project directories: {project_dirs}")
    
    if not success or not project_dirs:
        if "no_projects" in dir_list:
            print("📂 No projects found")
        else:
            print("📂 No valid beak project directories found")
        return {}
    
    print("🗂️  **Projects on Remote**")
    print()
    
    projects_info = {}
    
    for project_dir in project_dirs:
        if verbose:
            print(f"🔍 Scanning project: {project_dir}")
        
        # Check each operation type directly (similar to _scan_and_update_jobs)
        project_info = {}
        operations_to_check = ["search", "align", "taxonomy"]
        
        for operation in operations_to_check:
            op_dir = f"{central_tmp_dir}/{project_dir}/{operation}"
            
            # Check if operation directory exists using the proven method
            success, op_exists = execute_remote_command(sshProc, f"test -d ~/{op_dir} && echo 'exists' || echo 'not_exists'", timeout=5)
            op_exists_clean = _clean_terminal_output(op_exists)
            
            if verbose:
                print(f"   Checking {operation}: success={success}, raw='{op_exists}', clean='{op_exists_clean}'")
            
            if success and "exists" in op_exists_clean:
                if verbose:
                    print(f"   Found {operation} operation in {project_dir}")
                
                operation_info = _get_operation_info(sshProc, project_dir, operation, verbose)
                if operation_info:
                    project_info[operation] = operation_info
        
        if project_info:
            projects_info[project_dir] = project_info
            _display_project_info(project_dir, project_info)
    
    if not projects_info:
        print("📂 No valid projects with operations found")
    
    return projects_info


def _get_operation_info(sshProc, project_dir, operation, verbose=False):
    """
    Get information about a specific operation within a project.
    
    Args:
        sshProc: SSH connection
        project_dir (str): Project directory name
        operation (str): Operation type ('search', 'align', 'taxonomy')
        verbose (bool): Show debugging output
    
    Returns:
        dict: Operation information including counts and status
    """
    operation_path = f"~/beak_tmp/{project_dir}/{operation}"
    
    operation_info = {"type": operation, "status": "running"}
    
    try:
        if operation == "search":
            # Based on _scan_and_update_jobs, check for resultDB file for completion
            success, complete_check = execute_remote_command(sshProc, f"test -f {operation_path}/resultDB && echo 'complete' || echo 'incomplete'", timeout=5)
            complete_check_clean = _clean_terminal_output(complete_check)
            
            if verbose:
                print(f"   Search completion check: success={success}, raw='{complete_check}', clean='{complete_check_clean}'")
            
            if "complete" in complete_check_clean:
                operation_info["status"] = "completed"
                
                # Try to get hit count from results.tsv if it exists
                success, hit_count = execute_remote_command(sshProc, f"wc -l < {operation_path}/results.tsv 2>/dev/null || echo '0'", timeout=10)
                hit_count_clean = _clean_terminal_output(hit_count)
                
                if verbose:
                    print(f"   Hit count check: success={success}, raw='{hit_count}', clean='{hit_count_clean}'")
                
                if success and hit_count_clean.strip().isdigit():
                    count = max(0, int(hit_count_clean.strip()) - 1)  # Subtract header
                    operation_info["hit_count"] = count
                else:
                    operation_info["hit_count"] = "unknown"
                
                # Get database info from config
                success, config_content = execute_remote_command(sshProc, f"cat {operation_path}/config.json 2>/dev/null", timeout=10)
                if success and config_content:
                    config_clean = _clean_terminal_output(config_content)
                    try:
                        import json
                        config = json.loads(config_clean)
                        operation_info["database"] = config.get("database", "unknown")
                    except:
                        operation_info["database"] = "unknown"
                else:
                    operation_info["database"] = "unknown"
            else:
                operation_info["status"] = "running"
        
        elif operation == "align":
            # Based on _scan_and_update_jobs, check for aligned.fasta for completion
            success, complete_check = execute_remote_command(sshProc, f"test -f {operation_path}/aligned.fasta && echo 'complete' || echo 'incomplete'", timeout=5)
            complete_check_clean = _clean_terminal_output(complete_check)
            
            if verbose:
                print(f"   Align completion check: success={success}, raw='{complete_check}', clean='{complete_check_clean}'")
            
            if "complete" in complete_check_clean:
                operation_info["status"] = "completed"
                
                # Count sequences in aligned.fasta
                success, seq_count = execute_remote_command(sshProc, f"grep -c '^>' {operation_path}/aligned.fasta 2>/dev/null || echo '0'", timeout=10)
                seq_count_clean = _clean_terminal_output(seq_count)
                
                if verbose:
                    print(f"   Sequence count check: success={success}, raw='{seq_count}', clean='{seq_count_clean}'")
                
                if success and seq_count_clean.strip().isdigit():
                    operation_info["sequence_count"] = int(seq_count_clean.strip())
                else:
                    operation_info["sequence_count"] = "unknown"
                
                operation_info["output_file"] = "aligned.fasta"
            else:
                operation_info["status"] = "running"
        
        elif operation == "taxonomy":
            # Based on _scan_and_update_jobs, check for taxonomyResult for completion
            success, complete_check = execute_remote_command(sshProc, f"test -f {operation_path}/taxonomyResult && echo 'complete' || echo 'incomplete'", timeout=5)
            complete_check_clean = _clean_terminal_output(complete_check)
            
            if verbose:
                print(f"   Taxonomy completion check: success={success}, raw='{complete_check}', clean='{complete_check_clean}'")
            
            if "complete" in complete_check_clean:
                operation_info["status"] = "completed"
                
                # Try to count classifications from various possible output files
                for result_file in ['taxonomy_report.tsv', 'taxonomyResult', 'lca.tsv']:
                    success, line_count = execute_remote_command(sshProc, f"wc -l < {operation_path}/{result_file} 2>/dev/null || echo '0'", timeout=10)
                    line_count_clean = _clean_terminal_output(line_count)
                    
                    if verbose:
                        print(f"   Checking {result_file}: success={success}, raw='{line_count}', clean='{line_count_clean}'")
                    
                    if success and line_count_clean.strip().isdigit() and int(line_count_clean.strip()) > 0:
                        count = max(0, int(line_count_clean.strip()) - 1)  # Subtract header if present
                        operation_info["classification_count"] = count
                        break
                else:
                    operation_info["classification_count"] = "unknown"
            else:
                operation_info["status"] = "running"
    
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Error getting info for {operation}: {e}")
        operation_info["status"] = "error"
    
    return operation_info


def _display_project_info(project_name, project_info):
    """
    Display formatted project information.
    
    Args:
        project_name (str): Name of the project
        project_info (dict): Project information dictionary
    """
    print(f"**{project_name}**")
    
    for operation_type, operation_data in project_info.items():
        if operation_type == "search":
            hit_count = operation_data.get("hit_count", "unknown")
            database = operation_data.get("database", "unknown")
            status = operation_data.get("status", "unknown")
            
            if status == "completed":
                if hit_count == "unknown":
                    print(f"    🔍 Search results: {database} database \033[90m(completed)\033[0m")
                else:
                    print(f"    🔍 Search results: {database} database \033[90m({hit_count} hits)\033[0m")
            elif status == "running":
                print(f"    🔍 Search: {database} database \033[90m(running...)\033[0m")
            else:
                print(f"    🔍 Search: {database} database \033[90m(status: {status})\033[0m")
        
        elif operation_type == "align":
            seq_count = operation_data.get("sequence_count", "unknown")
            status = operation_data.get("status", "unknown")
            
            if status == "completed":
                if seq_count == "unknown":
                    print(f"    🧬 Alignment \033[90m(completed)\033[0m")
                else:
                    print(f"    🧬 Alignment \033[90m({seq_count} sequences)\033[0m")
            elif status == "running":
                print(f"    🧬 Alignment \033[90m(running...)\033[0m")
            else:
                print(f"    🧬 Alignment \033[90m(status: {status})\033[0m")
        
        elif operation_type == "taxonomy":
            class_count = operation_data.get("classification_count", "unknown")
            status = operation_data.get("status", "unknown")
            
            if status == "completed":
                if class_count == "unknown":
                    print(f"    🦠 Taxonomy classification \033[90m(completed)\033[0m")
                else:
                    print(f"    🦠 Taxonomy classification \033[90m({class_count} classifications)\033[0m")
            elif status == "running":
                print(f"    🦠 Taxonomy classification \033[90m(running...)\033[0m")
            else:
                print(f"    🦠 Taxonomy classification \033[90m(status: {status})\033[0m")
    
    print()  # Empty line between projects