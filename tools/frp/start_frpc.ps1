$ErrorActionPreference = "Stop"

$toolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$configPath = Join-Path $toolDir "frpc.toml"
$frpcPath = Join-Path $toolDir "frpc.exe"

if (-not (Test-Path -LiteralPath $configPath)) {
    throw "Missing $configPath. Copy frpc.example.toml to frpc.toml and set a new token first."
}

& $frpcPath -c $configPath
