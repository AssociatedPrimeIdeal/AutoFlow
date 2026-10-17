[CmdletBinding()]
param([string]$Destination = $PSScriptRoot)
$ErrorActionPreference = "Stop"
$Manifest = Get-Content -LiteralPath (Join-Path $PSScriptRoot "release-manifest.json") -Raw | ConvertFrom-Json
if (-not $Manifest.parts.Count) { Write-Host "This release already contains a complete ZIP."; exit 0 }
$Archive = Join-Path $Destination ([IO.Path]::GetFileName($Manifest.archive))
if (Test-Path -LiteralPath $Archive) { throw "Destination already exists: $Archive" }
foreach ($Part in $Manifest.parts) {
    $Path = Join-Path $PSScriptRoot ([IO.Path]::GetFileName($Part.file))
    if (-not (Test-Path -LiteralPath $Path)) { throw "Missing download: $($Part.file)" }
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Part.sha256) { throw "Checksum mismatch: $($Part.file)" }
}
$Staged = "$Archive.assembling"
$Output = [IO.File]::Open($Staged, [IO.FileMode]::CreateNew)
try {
    foreach ($Part in $Manifest.parts) {
        $PartStream = [IO.File]::OpenRead((Join-Path $PSScriptRoot ([IO.Path]::GetFileName($Part.file))))
        try { $PartStream.CopyTo($Output) } finally { $PartStream.Dispose() }
    }
} finally { $Output.Dispose() }
if ((Get-FileHash -LiteralPath $Staged -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Manifest.sha256) { throw "Joined ZIP checksum mismatch; incomplete file retained at $Staged" }
Move-Item -LiteralPath $Staged -Destination $Archive
Write-Host "Verified ZIP: $Archive"
Write-Host "Extract it and double-click AutoFlow.exe."
exit 0
