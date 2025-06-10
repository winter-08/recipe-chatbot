#!/usr/bin/env python
"""Start Phoenix server to view traces without running queries."""

import os
from pathlib import Path
import phoenix as px
from rich.console import Console

console = Console()

# Set up Phoenix to use the same file-based storage as the tracing script
storage_path = Path.home() / ".phoenix" / "datasets"
storage_path.mkdir(parents=True, exist_ok=True)
os.environ["PHOENIX_WORKING_DIR"] = str(storage_path)

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