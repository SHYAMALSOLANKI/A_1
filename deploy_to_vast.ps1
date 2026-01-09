# PowerShell Deployment Script for Vast.ai
param (
    [string]$IdentityFile = "", # Path to your private key (optional if in ssh-agent)
    [string]$HostIP = "192.165.134.28",
    [string]$Port = "22278",
    [string]$User = "root"
)

# Auto-detect local key if not provided
if ([string]::IsNullOrEmpty($IdentityFile) -and (Test-Path "keys\vast_ai_key")) {
    $IdentityFile = Resolve-Path "keys\vast_ai_key"
    Write-Host "Auto-detected local key: $IdentityFile" -ForegroundColor Yellow
}

$RemoteDest = "/root/A_1"
$ZipPath = ".\deploy_package.zip"

Write-Host "--- Starting Deployment to Vast.ai ($HostIP) ---" -ForegroundColor Cyan

# 1. Compress
if (Test-Path $ZipPath) { Remove-Item $ZipPath }
Write-Host "Compressing workspace (Excluding large data)..."
# Exclude training data, large models, and build artifacts to keep upload fast
$Exclusions = @(
    "deploy_package.zip", 
    "artifacts", 
    ".git", 
    ".venv", 
    "__pycache__", 
    "training_data",       # User request: Downloaded on remote
    "model.safetensors",   # Exclude large local models
    "drrl-*-output"        # Exclude local training outputs
)
Get-ChildItem -Path . -Exclude $Exclusions | Compress-Archive -DestinationPath $ZipPath
Write-Host "Compression complete."

# 2. Upload
$ScpArgs = @("-P", $Port)
if ($IdentityFile) { $ScpArgs += ("-i", $IdentityFile) }
$ScpArgs += ($ZipPath, "${User}@${HostIP}:/root/")

Write-Host "Uploading package... (This may take a minute)"
scp @ScpArgs

if ($LASTEXITCODE -ne 0) {
    Write-Error "SCP Failed. Please check your SSH Key and Connection."
    exit 1
}

# 3. Remote Setup & Run
$RemoteCmds = @(
    "apt-get update && apt-get install -y unzip dos2unix tmux",
    "rm -rf $RemoteDest",
    "unzip -o /root/deploy_package.zip -d $RemoteDest",
    "mkdir -p $RemoteDest/training_data", # Ensure mount point exists
    "cd $RemoteDest",
    "chmod +x run_native.sh",
    "dos2unix run_native.sh",
    "echo '--- Launching Training in Tmux Session training ---'",
    "tmux kill-session -t training 2>/dev/null || true", # Kill old session
    "tmux new-session -d -s training './run_native.sh'",
    "tmux attach -t training" # Attach so user sees output immediately
)
$RemoteScript = $RemoteCmds -join " && "

$SshArgs = @("-p", $Port, "-o", "StrictHostKeyChecking=no")
if ($IdentityFile) { $SshArgs += ("-i", $IdentityFile) }
$SshArgs += ("${User}@${HostIP}", $RemoteScript)

Write-Host "Executing remote build & launch..."
ssh @SshArgs
