# worshipper
ai stuff


### ssh
PowerShell:
    New-NetFirewallRule -DisplayName "WSL SSH" -Direction Inbound -Protocol TCP -LocalPort 2222 -RemoteAddress LocalSubnet -Action Allow

laptop:
    ssh -p 2222 <wsl_username>@WINDOWS_IP
    sftp -P 2222 <wsl_username>@WINDOWS_IP
