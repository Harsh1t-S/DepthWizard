<#
.SYNOPSIS
    Publish the DepthWizard backend to a Hugging Face Docker Space.

.DESCRIPTION
    Builds a clean orphan commit containing only what the Space needs, then
    force-pushes it to the Space's main branch. The working branch is restored
    afterwards even if the push fails.

    The Space must be configured with SDK "Docker" and free "CPU basic"
    hardware. The frontmatter in deploy/hf-space/README.md sets sdk and
    app_port; Space hardware itself is chosen in the Space settings UI.

.PARAMETER Remote
    Name of the git remote pointing at the Space. Add it once with:
      git remote add hf https://huggingface.co/spaces/<user>/<space>
#>
[CmdletBinding()]
param(
    [string]$Remote = "hf"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (git remote | Select-String -Quiet "^$Remote$")) {
    throw "Remote '$Remote' not found. Add it with: git remote add $Remote https://huggingface.co/spaces/<user>/<space>"
}

$current = (git rev-parse --abbrev-ref HEAD).Trim()

# Only tracked-file changes matter: the orphan commit adds an explicit file
# list, so stray untracked files in the workspace cannot leak into the Space.
# Uncommitted edits to tracked files would be silently lost by the checkout.
if ((git status --porcelain --untracked-files=no)) {
    throw "Tracked files have uncommitted changes. Commit or stash before deploying."
}

$branch = "hf-deploy-$(Get-Random)"
Write-Host "Building deploy commit on $branch (from $current)..." -ForegroundColor Cyan

try {
    git checkout --orphan $branch
    git reset

    # The Space needs the Dockerfile, the service packages, and a README whose
    # frontmatter declares the Docker SDK. Nothing else is copied, so frontend
    # assets and sample data never inflate the Space repository.
    Copy-Item deploy/hf-space/README.md README.md -Force
    git add README.md Dockerfile .dockerignore requirements.txt app.py backend/ ml/
    git commit -m "Deploy DepthWizard backend (Docker SDK, CPU)" | Out-Null

    Write-Host "Pushing to $Remote/main..." -ForegroundColor Cyan
    git push $Remote "${branch}:main" --force

    Write-Host "Deployed. Watch the build log in the Space 'Logs' tab." -ForegroundColor Green
}
finally {
    git checkout -f $current
    git branch -D $branch 2>$null | Out-Null
    git checkout -- README.md 2>$null | Out-Null
}
