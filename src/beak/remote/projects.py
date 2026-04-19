"""Project organization for grouping related beak jobs"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


class ProjectManager:
    """Manage local beak project organization"""

    def __init__(self, projects_dir: Optional[Path] = None):
        self.projects_dir = projects_dir or Path.home() / "beak_projects"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.projects_dir / ".index.json"

    def _get_index(self) -> Dict:
        """Load projects index"""
        if not self.index_file.exists():
            return {}
        with open(self.index_file) as f:
            return json.load(f)

    def _save_index(self, index: Dict):
        """Save projects index"""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)

    def organize_project(self,
                    job_ids: list,
                    project_name: str,
                    description: Optional[str] = None,
                    cleanup: bool = False) -> Path:
        """
        Organize multiple related jobs into a single coherent project

        Args:
            job_ids: List of job IDs to organize together
            project_name: Name for the organized project
            description: Optional project description
            cleanup: If True, remove original job directories after organizing

        Returns:
            Path to the organized project directory
        """
        from shutil import copytree, rmtree

        organized_dir = self.projects_dir / project_name
        if organized_dir.exists():
            raise ValueError(f"Project '{project_name}' already exists at {organized_dir}")

        organized_dir.mkdir(parents=True)
        index = self._get_index()

        job_metadata = []
        dirs_to_cleanup = []

        for job_id in job_ids:
            if job_id not in index:
                print(f"Warning: Job {job_id} not found in index, skipping")
                continue

            job_info = index[job_id]
            source_dir = Path(job_info['project_dir'])

            if not source_dir.exists():
                job_type = job_info.get('job_type', 'unknown')
                potential_dir = self.projects_dir / f"{job_type}_{job_id}"

                if potential_dir.exists():
                    print(f"Using standalone directory: {potential_dir.name}")
                    source_dir = potential_dir
                else:
                    old_organized = job_info.get('organized_project')
                    if old_organized:
                        old_organized_path = Path(job_info['project_dir'])
                        if old_organized_path.exists():
                            print(f"Found in organized project: {old_organized}/{old_organized_path.name}")
                            source_dir = old_organized_path
                        else:
                            print(f"Warning: Job {job_id} is organized in '{old_organized}' which doesn't exist")
                            if potential_dir.exists():
                                source_dir = potential_dir
                            else:
                                print(f"  Not found, skipping")
                                continue
                    else:
                        print(f"Warning: Project directory {source_dir} not found, skipping")
                        continue

            if not source_dir.exists():
                print(f"Warning: Could not locate files for job {job_id}, skipping")
                continue

            job_type = job_info.get('job_type', 'unknown')

            target_subdir = organized_dir / job_type
            counter = 1
            while target_subdir.exists():
                target_subdir = organized_dir / f"{job_type}_{counter}"
                counter += 1

            print(f"Copying {source_dir.name} -> {project_name}/{target_subdir.name}")
            copytree(source_dir, target_subdir)

            if cleanup:
                dirs_to_cleanup.append(source_dir)

            job_metadata.append({
                'job_id': job_id,
                'job_type': job_type,
                'original_name': job_info.get('name'),
                'subdirectory': target_subdir.name,
                'created': job_info.get('created'),
            })

            index[job_id]['project_dir'] = str(target_subdir)
            index[job_id]['organized_project'] = project_name

        readme_content = f"""# {project_name}

    {description or 'No description provided'}

    ## Jobs Included

    """
        for meta in job_metadata:
            readme_content += f"""
    ### {meta['job_type'].upper()}: {meta['original_name']}
    - **Job ID**: `{meta['job_id']}`
    - **Directory**: `{meta['subdirectory']}/`
    - **Created**: {meta.get('created', 'Unknown')}
    """

        readme_file = organized_dir / "README.md"
        with open(readme_file, 'w') as f:
            f.write(readme_content)

        project_metadata = {
            'project_name': project_name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'jobs': job_metadata
        }

        metadata_file = organized_dir / "project_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(project_metadata, f, indent=2)

        self._save_index(index)

        if cleanup and dirs_to_cleanup:
            print(f"\nCleaning up {len(dirs_to_cleanup)} original job directories...")
            for dir_path in dirs_to_cleanup:
                print(f"  Removing {dir_path.name}")
                rmtree(dir_path)
            print("✓ Cleanup complete")

        print(f"\n✓ Organized project created: {project_name}")
        print(f"  Location: {organized_dir}")
        print(f"  Jobs: {len(job_metadata)}")
        print(f"\nProject structure:")

        for item in sorted(organized_dir.iterdir()):
            if item.is_dir():
                print(f"  {item.name}/")
                for subitem in sorted(item.iterdir()):
                    print(f"    └── {subitem.name}")
            else:
                print(f"  {item.name}")

        return organized_dir

    def list_projects(self) -> pd.DataFrame:
        """List all projects (both individual jobs and organized projects)"""
        index = self._get_index()

        if not index:
            return pd.DataFrame()

        projects = []
        for job_id, info in index.items():
            projects.append({
                'job_id': job_id,
                'name': info.get('name'),
                'job_type': info.get('job_type'),
                'created': info.get('created'),
                'organized_project': info.get('organized_project', None),
                'path': info.get('project_dir')
            })

        df = pd.DataFrame(projects)
        if not df.empty:
            df = df.sort_values('created', ascending=False)

        return df

    def list_organized_projects(self) -> pd.DataFrame:
        """List all organized projects"""
        projects = []

        for item in self.projects_dir.iterdir():
            if not item.is_dir():
                continue

            metadata_file = item / "project_metadata.json"
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                projects.append({
                    'project_name': metadata['project_name'],
                    'description': metadata.get('description', ''),
                    'created': metadata['created_at'],
                    'num_jobs': len(metadata['jobs']),
                    'job_types': ', '.join(set(j['job_type'] for j in metadata['jobs'])),
                    'path': str(item)
                })

        if not projects:
            return pd.DataFrame()

        return pd.DataFrame(projects).sort_values('created', ascending=False)

    def get_project_info(self, project_name: str) -> Dict:
        """Get detailed information about an organized project"""
        project_dir = self.projects_dir / project_name
        metadata_file = project_dir / "project_metadata.json"

        if not metadata_file.exists():
            raise ValueError(f"Project '{project_name}' not found or not an organized project")

        with open(metadata_file) as f:
            return json.load(f)
