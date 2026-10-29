"""Execute the teaching notebooks in order using the selected environment."""
from pathlib import Path  # Locate notebooks consistently from any working directory.
import asyncio  # Select the event loop supported by Windows notebook communication.
import sys  # Detect Windows without assuming every reader uses it.
import nbformat  # Read and save normal Jupyter notebook files.
from nbclient import NotebookClient  # Execute each code cell and fail on errors.

if sys.platform == "win32":  # Windows needs selector support for notebook messaging.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # Avoid the extra event loop compatibility thread.
root = Path(__file__).resolve().parents[1]  # Find the shared project directory.
for path in sorted((root / "notebooks").glob("*.ipynb")):  # Respect the teaching sequence.
    print(path.name, flush=True)  # Report which notebook is currently running.
    notebook = nbformat.read(path, as_version=4)  # Preserve explanations and existing metadata.
    NotebookClient(notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()  # Run cells in order without changing the experimental protocol.
    nbformat.write(notebook, path)  # Save outputs so readers can inspect results without a dataset download.
