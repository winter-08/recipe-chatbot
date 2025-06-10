#!/usr/bin/env python
"""Serve Phoenix UI with persistent storage.

This script starts the Phoenix server with the same storage location
used by the tracing scripts, allowing you to view persisted traces.

Usage:
    uv run python scripts/phoenix_serve.py
"""

import os
from pathlib import Path
import phoenix as px
from rich.console import Console

console = Console()

# Set up Phoenix to use the same file-based storage as the tracing script
storage_path = Path.home() / ".phoenix" / "datasets"
storage_path.mkdir(parents=True, exist_ok=True)
os.environ["PHOENIX_WORKING_DIR"] = str(storage_path)

console.print(f"[yellow]Using Phoenix storage at: {storage_path}[/yellow]")

# Launch Phoenix app with persistent storage
app = px.launch_app()

console.print("[bold green]Phoenix UI is running at: http://localhost:6006[/bold green]")
console.print("[yellow]Press Ctrl+C to stop the server[/yellow]")

try:
    # Keep the server running
    import time
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    console.print("\n[yellow]Shutting down Phoenix server...[/yellow]")