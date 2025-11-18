import numpy as np
import subprocess
import sys
import tempfile
import os
import shutil
from multiprocessing import shared_memory

def run_cotracker_blocking(frames, queries, online_mode, start=None, end=None):
    # Actual model running as in a subprocess to avoid any dll dependency problems with napari and qt
    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp()
        frames_path = os.path.join(tmpdir, 'frames.npy')
        queries_path = os.path.join(tmpdir, 'queries.npy')
        output_path = os.path.join(tmpdir, 'tracks.npy')
        
        np.save(frames_path, frames) 
        np.save(queries_path, queries)
        
        worker_script = os.path.join(os.path.dirname(__file__), '_subprocess.py')
        
        cmd = [sys.executable, worker_script, frames_path, queries_path, output_path]
        
        if online_mode:
            cmd.append('--online')
        if start is not None:
            cmd.extend(['--start', str(start)])
        if end is not None:
            cmd.extend(['--end', str(end)])
        
        # Env cleanup
        env = os.environ.copy()
        
        # Run in completely isolated process
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=os.path.dirname(worker_script),
            check=True,
            timeout=300, # 5 minutes
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"CoTracker failed:\n{result.stderr}")
        
        tracks = np.load(output_path)
        n_timepoints, n_tracks, n_coords = tracks.shape
        timepoints = np.arange(n_timepoints)
        timepoints = np.repeat(timepoints, n_tracks)
        coords = tracks.reshape(-1, n_coords)
        result = np.column_stack([timepoints, coords])
    
        return result
    
    finally:
        # Cleanup temp
        if tmpdir and os.path.exists(tmpdir):
            try:
                shutil.rmtree(tmpdir)
            except Exception as e:
                print(f"Could not delete temp dir {tmpdir}: {e}")