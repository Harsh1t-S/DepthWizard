[CmdletBinding()]
param(
    [string]$BindAddress = "127.0.0.1",
    [ValidateRange(1, 65535)]
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FrontendRoot = Join-Path $RepoRoot "frontend"

if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot "package.json"))) {
    throw "frontend/package.json was not found."
}
if (-not (Test-Path -LiteralPath (Join-Path $FrontendRoot "node_modules"))) {
    throw "Frontend dependencies are missing. Run 'npm install' in frontend first."
}
$NpmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $NpmCommand) {
    throw "npm.cmd was not found. Install Node.js and npm first."
}

Set-Location $FrontendRoot
& $NpmCommand.Source run dev -- --host $BindAddress --port $Port
exit $LASTEXITCODE

