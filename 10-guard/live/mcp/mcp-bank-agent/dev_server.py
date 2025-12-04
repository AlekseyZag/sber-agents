#!/usr/bin/env python3
"""
Development server with auto-reload for MCP Bank Agent

Watches for file changes and automatically restarts the server.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("❌ watchdog not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler


class ReloadHandler(FileSystemEventHandler):
    """Handler for file system events that restarts the server on .py file changes"""
    
    def __init__(self):
        self.process = None
        self.restart()
    
    def restart(self):
        """Restart the server process"""
        if self.process:
            print("🛑 Stopping server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        print("🚀 Starting server...")
        self.process = subprocess.Popen(
            [sys.executable, "server.py"],
            cwd=os.getcwd()
        )
    
    def on_modified(self, event):
        """Called when a file is modified"""
        if event.src_path.endswith('.py') and not event.src_path.endswith('dev_server.py'):
            print(f"\n🔄 File changed: {os.path.basename(event.src_path)}")
            print("   Reloading server...")
            self.restart()


if __name__ == "__main__":
    print("=" * 60)
    print("🔄 MCP Bank Agent - Development Mode (Auto-reload)")
    print("=" * 60)
    print("📍 Server: http://localhost:8000/mcp")
    print("📝 Watching for .py file changes...")
    print("⏹️  Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    # Create observer
    observer = Observer()
    handler = ReloadHandler()
    
    # Watch current directory recursively
    observer.schedule(handler, '.', recursive=True)
    observer.start()
    
    try:
        # Keep running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        observer.stop()
        if handler.process:
            handler.process.terminate()
            try:
                handler.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                handler.process.kill()
    
    observer.join()
    print("✅ Server stopped")

