import os
import psutil
import torch
import signal

def cleanup_processes():
    """Safely clean up only our own Python processes and GPU memory"""
    try:
        # Get current user's processes only
        current_user = os.getenv('USER')
        current_pid = os.getpid()  # Get our own process ID
        parent_pid = os.getppid()  # Get parent process ID
        
        # Find Python processes owned by current user
        for proc in psutil.process_iter(['pid', 'name', 'username']):
            try:
                # Only kill processes that:
                # 1. Are owned by current user
                # 2. Are Python processes
                # 3. Are not the current cleanup script or its parent
                if (proc.info['username'] == current_user and 
                    'python' in proc.info['name'].lower() and 
                    proc.pid != current_pid and 
                    proc.pid != parent_pid):
                    
                    # Check if process is using our ports
                    try:
                        connections = proc.connections()
                        for conn in connections:
                            if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port'):
                                if conn.laddr.port in range(29500, 29510):
                                    print(f"Found training process {proc.pid} using port {conn.laddr.port}")
                                    # Try graceful termination first
                                    proc.terminate()
                                    try:
                                        proc.wait(timeout=3)  # Wait up to 3 seconds
                                    except psutil.TimeoutExpired:
                                        # If process doesn't terminate gracefully, force kill
                                        print(f"Force killing process {proc.pid}")
                                        proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Clear CUDA memory if available
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.set_device(i)
                    torch.cuda.empty_cache()
                print("Cleared CUDA memory")
            except Exception as e:
                print(f"Warning: Could not clear CUDA memory: {e}")

        print("Cleanup completed")
        
    except Exception as e:
        print(f"Warning during cleanup: {e}")
        # Continue even if cleanup has issues
        pass

if __name__ == "__main__":
    cleanup_processes() 