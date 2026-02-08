param(
    [string]$remoteUrl = "https://github.com/Homssalomssa/bvmt-sentiment-analysis.git",
    [string]$subfolder = "bvmt-forecasting",
    [string]$branch = "independant"
)

Write-Host "Remote: $remoteUrl"
Write-Host "Subfolder: $subfolder"
Write-Host "Branch: $branch"

# Initialize repo if needed
try { git rev-parse --is-inside-work-tree > $null 2>&1 } catch { }
if ($LASTEXITCODE -ne 0) {
    git init
    Write-Host "Initialized new git repo"
}

# Create branch
git checkout -b $branch

# Ensure subfolder exists
if (-not (Test-Path $subfolder)) { New-Item -ItemType Directory -Path $subfolder | Out-Null }

# Determine tracked files
$tracked = git ls-files

# Exclude this script and push scripts
$exclude = @('move_and_push.ps1','push_subfolder.bat','push_subfolder.sh')

if ($tracked) {
    Write-Host "Moving tracked files into $subfolder..."
    foreach ($f in $tracked) {
        if ($exclude -contains $f) { continue }
        $dest = Join-Path $subfolder $f
        $destdir = Split-Path $dest
        if (!(Test-Path $destdir)) { New-Item -ItemType Directory -Path $destdir -Force | Out-Null }
        try {
            git mv "$f" "$dest" 2>$null
        } catch {
            Move-Item -Force -Path $f -Destination $dest
        }
    }
} else {
    Write-Host "No tracked files found — moving working files into $subfolder"
    Get-ChildItem -Force -File | Where-Object { $exclude -notcontains $_.Name -and $_.Name -ne '.git' } | ForEach-Object {
        $dest = Join-Path $subfolder $_.Name
        Move-Item -Force -Path $_.FullName -Destination $dest
    }
}

# Add and commit
git add -A
try { git commit -m "Move project into subfolder $subfolder" } catch { Write-Host "Commit failed or nothing to commit" }

# Add remote and push
git remote add target $remoteUrl 2>$null || Write-Host "Remote 'target' already exists"
Write-Host "Pushing to remote 'target' branch $branch (you may be prompted for credentials)..."
git push target $branch

Write-Host "Done. If push failed due to auth, please run the script locally with appropriate credentials."