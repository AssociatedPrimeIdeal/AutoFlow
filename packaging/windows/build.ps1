[CmdletBinding()]
param(
    [string]$PythonBin = "python",
    [string]$EnvironmentDirectory = "",
    [string]$OutputDirectory = "",
    [string]$ModelsArchive = "",
    [string]$DocumentationDirectory = "",
    [string]$Version = "",
    [switch]$SkipArchive,
    [switch]$SkipDependencyInstall
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).ProviderPath
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $ProjectRoot "release" }
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
if (-not $EnvironmentDirectory) {
    $EnvironmentDirectory = Join-Path ([IO.Path]::GetTempPath()) "autoflow-windows-build"
}
if (-not $Version) {
    $Head = (& git -C $ProjectRoot rev-parse --short=8 HEAD).Trim()
    $Version = "dev-$Head"
}
if ($Version -notmatch '^[a-zA-Z0-9][a-zA-Z0-9._-]*$') { throw "Invalid package version" }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$EnvironmentDirectory = [IO.Path]::GetFullPath($EnvironmentDirectory)
$BuildPython = Join-Path $EnvironmentDirectory "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $BuildPython)) {
    & $PythonBin -m venv $EnvironmentDirectory
    if ($LASTEXITCODE -ne 0) { throw "Could not create the Windows build environment" }
}
& $BuildPython -c "import sys,struct; assert sys.version_info[:2] == (3,11) and struct.calcsize('P') == 8, 'Windows x64 Python 3.11 is required'"
if ($LASTEXITCODE -ne 0) { throw "Unsupported build interpreter" }
if (-not $SkipDependencyInstall) {
    & $BuildPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) { throw "Build tools could not be installed" }
    & $BuildPython -m pip install "torch==2.13.0" -c (Join-Path $PSScriptRoot "constraints.txt") --index-url https://download.pytorch.org/whl/cu130
    if ($LASTEXITCODE -ne 0) { throw "CUDA PyTorch could not be installed" }
    & $BuildPython -m pip install -r (Join-Path $PSScriptRoot "requirements.txt") -c (Join-Path $PSScriptRoot "constraints.txt")
    if ($LASTEXITCODE -ne 0) { throw "Full application dependencies could not be installed" }
}

# Documentation build tools stay outside the application runtime environment.
if (-not $DocumentationDirectory) {
    $DocsEnvironment = Join-Path $OutputDirectory "build-docs-env"
    $DocsPython = Join-Path $DocsEnvironment "Scripts/python.exe"
    if (-not (Test-Path -LiteralPath $DocsPython)) {
        & $BuildPython -m venv $DocsEnvironment
        if ($LASTEXITCODE -ne 0) { throw "Could not create the documentation build environment" }
    }
    & $DocsPython -m pip install -r (Join-Path $PSScriptRoot "../docs/requirements.txt")
    if ($LASTEXITCODE -ne 0) { throw "Documentation build tools could not be installed" }
    $DocumentationDirectory = Join-Path $OutputDirectory "build-docs"
    & $DocsPython (Join-Path $PSScriptRoot "../docs/build_offline.py") --repo $ProjectRoot --output $DocumentationDirectory
    if ($LASTEXITCODE -ne 0) { throw "Offline documentation build failed" }
}
$DocumentationDirectory = (Resolve-Path -LiteralPath $DocumentationDirectory).ProviderPath
if (-not (Test-Path -LiteralPath (Join-Path $DocumentationDirectory "offline-manifest.json"))) {
    throw "DocumentationDirectory must contain verified offline documentation"
}

# Build each optional package from this checkout, including local edits.
$SourceStage = Join-Path $OutputDirectory "build-sources"
New-Item -ItemType Directory -Path $SourceStage -Force | Out-Null
foreach ($Project in @("4DFlow_Dicom2H5", "PUDIP-Flow", "GUST-Flow", "SpatioTemporalLabeler")) {
    $Source = Join-Path $ProjectRoot "third_party/$Project"
    if (-not (Test-Path -LiteralPath (Join-Path $Source "pyproject.toml"))) { throw "Missing source submodule: $Project" }
    $Target = [IO.Path]::GetFullPath((Join-Path $SourceStage $Project))
    if (-not $Target.StartsWith($SourceStage + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe source staging path" }
    if (Test-Path -LiteralPath $Target) { Remove-Item -LiteralPath $Target -Recurse -Force }
    New-Item -ItemType Directory -Path $Target | Out-Null
    Copy-Item -LiteralPath (Join-Path $Source "src") -Destination $Target -Recurse
    foreach ($Name in @("pyproject.toml", "README.md", "LICENSE", "LICENSE.txt")) {
        if (Test-Path -LiteralPath (Join-Path $Source $Name)) { Copy-Item -LiteralPath (Join-Path $Source $Name) -Destination $Target }
    }
    & $BuildPython -m pip install --no-deps --no-build-isolation --upgrade $Target
    if ($LASTEXITCODE -ne 0) { throw "Could not install current $Project sources" }
}

if (-not $ModelsArchive) { $ModelsArchive = Join-Path $ProjectRoot "build/windows-models/autoflow-models.zip" }
if (-not (Test-Path -LiteralPath $ModelsArchive)) {
    $Lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot "models.lock.json") -Raw | ConvertFrom-Json
    $ModelsArchive = Join-Path $OutputDirectory $Lock.asset
    Invoke-WebRequest -Uri $Lock.url -OutFile $ModelsArchive
}
$Lock = Get-Content -LiteralPath (Join-Path $PSScriptRoot "models.lock.json") -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $ModelsArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Lock.sha256) {
    throw "Model archive checksum does not match packaging/windows/models.lock.json"
}
$ModelsStage = Join-Path $OutputDirectory "build-models"
if (Test-Path -LiteralPath $ModelsStage) {
    $ResolvedModels = (Resolve-Path -LiteralPath $ModelsStage).ProviderPath
    if (-not $ResolvedModels.StartsWith($OutputDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe model staging path" }
    Remove-Item -LiteralPath $ResolvedModels -Recurse -Force
}
Expand-Archive -LiteralPath $ModelsArchive -DestinationPath $ModelsStage
$PackageName = "AutoFlow-$Version-windows-x64"
$PackageRoot = [IO.Path]::GetFullPath((Join-Path $OutputDirectory $PackageName))
if (-not $PackageRoot.StartsWith($OutputDirectory + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe package output path" }
if (Test-Path -LiteralPath $PackageRoot) { Remove-Item -LiteralPath $PackageRoot -Recurse -Force }
& $BuildPython (Join-Path $PSScriptRoot "assemble.py") stage --repo $ProjectRoot --destination $PackageRoot --models (Join-Path $ModelsStage "models") --version $Version --docs-site $DocumentationDirectory
if ($LASTEXITCODE -ne 0) { throw "Portable assembly failed" }
$Compiler = Join-Path $env:WINDIR "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
& $Compiler /nologo /target:winexe /platform:x64 /reference:System.Windows.Forms.dll "/out:$PackageRoot/AutoFlow.exe" (Join-Path $PSScriptRoot "Launcher.cs")
if ($LASTEXITCODE -ne 0) { throw "GUI launcher compilation failed" }
& (Join-Path $PackageRoot "_runtime/python.exe") (Join-Path $PackageRoot "app/portable_check.py") --gui --output (Join-Path $PackageRoot "build-check.json")
if ($LASTEXITCODE -ne 0) { throw "Portable runtime verification failed; see build-check.json" }
if (-not $SkipArchive) {
    & $BuildPython (Join-Path $PSScriptRoot "assemble.py") archive --destination $PackageRoot --output (Join-Path $OutputDirectory "artifacts")
    if ($LASTEXITCODE -ne 0) { throw "Portable archive creation failed" }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "Join-Portable.ps1") -Destination (Join-Path $OutputDirectory "artifacts/Join-Portable.ps1") -Force
}
Write-Host "Portable directory: $PackageRoot"
Write-Host "Double-click AutoFlow.exe; models are in the separate models directory."
