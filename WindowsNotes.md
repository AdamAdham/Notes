# Windows-Notes
# Python

### Check Installed Python versions and the Default
```cmd
py -0
```

### Run a Specific Version Temporarily
You can specify the version directly when running Python:
```cmd
py -3.9   # Runs Python 3.9  
py -3.12  # Runs Python 3.12
```

## Start/Restart/Stop PostgreSQL service

1. Open the "Run" dialog by pressing Win + R.
2. Type services.msc and press Enter to open the Services Manager.
3. In the Services Manager, locate the PostgreSQL service. It's usually named something like
   "postgresql-x64-15" or similar, depending on your PostgreSQL version and architecture.
   Right-click on the PostgreSQL service and select "Stop" from the context menu.

## To check for where a version is from

```
where java
```

Outputs the paths that java is using in order of priority

# Environmental Variables

### Check Environmental Variables

In powershell

```
Get-ChildItem -Path Env:\
```

# Managing Node.js

### Listing currently installed versions

```
nvm ls
```

### Change current Node.js version

```
nvm use <version>
```
