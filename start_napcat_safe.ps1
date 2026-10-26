param(
    [Parameter(Mandatory=$true)][string]$NapCatDir,
    [Parameter(Mandatory=$true)][string]$UserDataDir,
    [switch]$CheckOnly
)
$ErrorActionPreference='Stop'
$separator=[IO.Path]::DirectorySeparatorChar
$base=[IO.Path]::GetFullPath($NapCatDir).TrimEnd($separator)
$data=[IO.Path]::GetFullPath($UserDataDir)
if (-not $data.StartsWith($base+$separator,[StringComparison]::OrdinalIgnoreCase)) { throw 'User data must stay inside this NapCat installation.' }
$qq=[IO.Path]::GetFullPath((Join-Path $base 'qq_extract\Files\QQ.exe'))
$runtime=Join-Path $base 'runtime'
foreach($file in @($qq,(Join-Path $base 'launch.ps1'),(Join-Path $runtime 'NapCatWinBootHook.dll'),(Join-Path $runtime 'napcat.mjs'))) {
    if (-not [IO.File]::Exists($file)) { throw 'NapCat installation is incomplete.' }
}
# This check is not permission to stop anything: there is NO stop/kill here.
$running=Get-CimInstance Win32_Process -Filter "Name='QQ.exe'" | Where-Object {
    $_.ExecutablePath -and [String]::Equals($_.ExecutablePath,$qq,[StringComparison]::OrdinalIgnoreCase) -and $_.CommandLine -notmatch '\s--type[=\s]'
}
if($running) { Write-Output 'NapCat already running; retained.'; exit 0 }
if($CheckOnly) { Write-Output 'Check passed; no process started or stopped.'; exit 0 }
$tmp=Join-Path $base 'tmp'
[IO.Directory]::CreateDirectory($tmp) | Out-Null
$env:TEMP=$tmp; $env:TMP=$tmp
# These variables affect this child process only. No registry or global folders change.
$env:APPDATA=Split-Path -Parent $data
$env:NAPCAT_PATCH_PACKAGE=Join-Path $runtime 'qqnt.json'
$env:NAPCAT_LOAD_PATH=Join-Path $runtime 'loadNapCat.js'
$env:NAPCAT_INJECT_PATH=Join-Path $runtime 'NapCatWinBootHook.dll'
$env:NAPCAT_LAUNCHER_PATH=Join-Path $runtime 'NapCatWinBootMain.exe'
$env:NAPCAT_MAIN_PATH=Join-Path $runtime 'napcat.mjs'
& (Join-Path $base 'launch.ps1') -QQExe $qq -HookDll (Join-Path $runtime 'NapCatWinBootHook.dll') `
    -WorkDir (Join-Path $base 'qq_extract\Files') -LogFile (Join-Path $base 'napcat.log') `
    -ExtraArgs ('--no-sandbox --user-data-dir="'+$data+'"')
