# Start the teaching notebooks with the project environment.
$ErrorActionPreference = 'Stop' # Report a missing environment rather than silently using another Python.
$projectRoot = Split-Path -Parent $PSScriptRoot # Locate the project containing this script.
Set-Location -LiteralPath $projectRoot # Make relative data paths resolve inside the project.
$pythonPath = Join-Path $projectRoot '.conda/python.exe' # Select the environment created from environment.yml.
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the local .conda environment first; see README.md.' } # Explain the required setup.
$env:PATH = (Join-Path $projectRoot '.conda') + ';' + (Join-Path $projectRoot '.conda/Library/bin') + ';' + (Join-Path $projectRoot '.conda/Scripts') + ';' + $env:PATH # Expose Windows numerical library dependencies.
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8' # Make deterministic CUDA matrix multiplication available.
& $pythonPath -m jupyterlab --ip=127.0.0.1 --port=8890 --ServerApp.root_dir=$projectRoot # Open a local authenticated Jupyter server.
