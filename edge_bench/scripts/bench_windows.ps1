# Build (if needed) and benchmark mnn_bench on Windows.
param(
    [string]$Binary = "build\Release\mnn_bench.exe",
    [string]$BuildDir = "build",
    [string]$ModelDir = "data\models_mnn",
    [string]$Backend = "CPU",
    [int]$Threads = 4,
    [string]$DatasetJson = "",
    [string]$BaseDirs = "rgb,rgb_with_bg",
    [int]$MaxImages = 1000,
    [int]$Warmup = 10,
    [int]$Repeat = 1000,
    [string]$OutputJson = "",
    [string]$CMakeArgs = "",
    [switch]$Force,
    [switch]$Overwrite
)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if ([string]::IsNullOrEmpty($OutputJson)) {
    $OutputJson = "output\result_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
}

if ($Force -or -not (Test-Path $Binary)) {
    Write-Host "Building mnn_bench..."
    $CMakeArgArray = @()
    if (-not [string]::IsNullOrEmpty($CMakeArgs)) {
        $CMakeArgArray = $CMakeArgs -split '\s+'
    }
    cmake -S . -B $BuildDir @CMakeArgArray
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    cmake --build $BuildDir --config Release -j 16
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if (-not (Test-Path $Binary)) {
    Write-Error "Binary not found: $Binary"
    exit 1
}

if ((Test-Path $OutputJson) -and -not $Overwrite) {
    Write-Host "Skip: output exists ($OutputJson); use -Overwrite to overwrite."
    exit 0
}

# Run each model in a separate process. The AMD OpenCL driver can TDR when
# certain models (e.g. swin_tiny -> vit_b) run back-to-back in one process.
# Per-model isolation avoids that cross-model GPU state interaction and keeps
# one crashing model from killing the whole benchmark run.
$ModelFiles = @()
if (Test-Path $ModelDir) {
    $ModelFiles = Get-ChildItem -Path $ModelDir -Recurse -Filter *.mnn -File |
        Sort-Object FullName | ForEach-Object { $_.FullName }
}

if ($ModelFiles.Count -eq 0) {
    Write-Error "No .mnn models found in: $ModelDir"
    exit 1
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("mnn_bench_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

$CollectedModels = @()
$FailedModels = @()
$ModelIndex = 0

foreach ($ModelPath in $ModelFiles) {
    $ModelIndex++
    $TempJson = Join-Path $TempDir ("model_{0:D3}.json" -f $ModelIndex)
    $RunArgs = @("--model", $ModelPath, "--backend", $Backend, "--threads", $Threads, "--warmup", $Warmup, "--repeat", $Repeat)
    if (-not [string]::IsNullOrEmpty($DatasetJson)) {
        $RunArgs += @("--dataset-json", $DatasetJson, "--base-dirs", $BaseDirs, "--max-images", $MaxImages)
    }
    $RunArgs += @("--output-json", $TempJson)

    Write-Host ("[{0}/{1}] {2}" -f $ModelIndex, $ModelFiles.Count, (Split-Path $ModelPath -Leaf))
    & $Binary @RunArgs @args
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Model failed (exit $LASTEXITCODE): $ModelPath"
        $FailedModels += $ModelPath
        continue
    }
    if (Test-Path $TempJson) {
        $Parsed = Get-Content $TempJson -Raw | ConvertFrom-Json
        if ($null -ne $Parsed.models) {
            $CollectedModels += $Parsed.models
        }
    }
}

$Result = [ordered]@{
    backend = $Backend
    threads = $Threads
    warmup = $Warmup
    repeat = $Repeat
    max_images = $MaxImages
    models = $CollectedModels
}
$Result | ConvertTo-Json -Depth 10 | Set-Content -Path $OutputJson -Encoding UTF8

Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue

if ($FailedModels.Count -gt 0) {
    Write-Warning ("Completed with {0} failed model(s): {1}" -f $FailedModels.Count, ($FailedModels -join ', '))
} else {
    Write-Host "Result: $OutputJson"
}
