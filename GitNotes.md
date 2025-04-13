# Git Notes
## Tracking
### Stop tracking after already committed
To stop tracking something that’s already been committed in Git and make sure it’s not tracked again (like a file or folder you’ve just added to .gitignore), follow these steps:

🧼 1. Add it to .gitignore
```gitignore
**/pycache
```
🗑 2. Remove it from Git’s tracking, but not from disk
Use the --cached flag to untrack the file/folder without deleting it locally:
```bash
git rm -r --cached data/stocks/raw/
```
✅ 3. Commit the removal
```bash
git commit -m "Stop tracking data/stocks/raw/"
```
## Cloning

**Decription**: Getting downloading the repository </br>
**Code**: git clone -b <ins>_branch_</ins> <ins>_link of repository_</ins> </br>
**Code**: git clone <ins>_link of repository_</ins>

## Status

**Decription**: Knowing which files are modified, not modified or untracked and not saved in a commit yet</br>
**Code**: git status

## Add

**Decription**: Track an untracked file </br>
**Code**: git add <ins>_name of file_</ins> / <ins>_._</ins> </br>
**Notes**: . is to track all files in current directory

## Commiting

**Decription**: Saving the project LOCALLY </br>
**Code**: git commit -m "<ins>_Commit message_</ins>" -m "<ins>_Commit Description_</ins>" </br>
**Notes**: Commit message is mandatory, Commit descripton is optional, smth to do with what and why you committed

## Push

**Decription**: Uploading to remote repository where my project is hosted </br>
**Notes**: Origin is the location of the git repository, master is the branch </br>
**Shortcut**: set smth called upstream so we use git push only by </br>

```GIT
git push (-u) origin master
```

## SSH Keys

**Decription**:Used to prove you are the owner of github account/ Connect local machine to github account </br>
**Code**: ssh-keygen -t rsa -b 4096 -C "<ins>_Github email address_</ins>" </br>
**Notes**: Code generates SSH key, -t: type of encryption, -b: strength of encryption </br>
There is a file for key and key.pub, key.pub is going to be uploaded to github interface (pub stands for public) </br>
Testkey is private (dont show to anyone) this key shows github you are the person that generated that public key </br>
Last step we just need to do is that let the local git cmd line know about the ssh key (search on internet)

## Start a git repository

**Decription**: create repository locally </br>
**Code**: git init </br>
**Notes**: Error will occur if we push (because was not cloned)

## Connect to a Repository on Github

**Decription**: To be able to push and pull </br>
**Code**: git remote add origin <ins>_link_</ins>

## Commiting

**Decription**: Saving the project LOCALLY </br>
**Code**: git commit -m "<ins>_Commit message_</ins>" -m "<ins>_Commit Description_</ins>" </br>
**Notes**: Commit message is mandatory, Commit descripton is optional, smth to do with what and why you committed

## See previous commits

```git
git log
```

stop by pressing q

## Undo Commit

**Decription**: Switch between branches </br>
**Code**: git reset HEAD~1 </br> / git reset <ins> _hash of commit_</ins> / git reset --hard <ins>_hash of commit_</ins>

**Notes**:

1. HEAD: Last commit
2. ~1: Go back 1 commit
3. To hash of commit: Resets any change after that commit
4. Hard: Doesnt only unstage, it completely removes anything after that commit

## Log

**Decription**: List of commits in reverse chronological order </br>
**Code**: git log </br>

## Basic steps for uploading repository

1. git add .
2. git commit -m "<ins>_Commit message_</ins>"
3. git push origin master
4. pull (not a pull request) (if a review or permission is needed)

## Branches

**Decription**: See branches of repository </br>
**Notes**: The one with astrisk is one currently on

```GIT
git branch
```

### Checkout

**Decription**: Switch between branches </br>
**Code**: git checkout <ins>_name of branch_</ins> </br>

### Get branches from the remote

```GIT
git fetch
```

### List all branches (local and remote)

```GIT
git branch -a
```

### Create Branch

**Decription**: Switch working space from current branch to another new one</br>

```GIT
git checkout -b name_of_branch
```

### Push Created Branch Remote

```git
git push -u origin new_branch_name
```

### Delete Branch

**Decription**: self explanatory </br>

```GIT
git branch -d name_of_branch
```

### Rename Branch

```git
git branch -m "name"
```

### Reset local main branch to match remote main branch:

This command will make your local main branch identical to the origin/main branch, discarding any local commits that aren't in the remote branch. Be cautious when using --hard, as it will overwrite changes in your working directory that haven’t been committed.

```git
git reset --hard origin/main
```

## Difference

**Decription**: See difference between current and specified branch </br>
**Code**: git diff <ins>_name of branch_</ins>/_ <ins>nothing </ins>to see diff of last commit_

## Pull Request

**Decription**: Code from pulled in from another branch </br>
**Code**: Did on github </br>

## Merge Conflicts

**Decription**: When same branch is being changed at same time </br>
**Code**: do on vs code/github

## Unstage

**Decription**: Make git not take into consuderation specified file</br>
**Code**: git reset <ins>_name of file_</ins> </br>

## Notes

When working on a branch every once in a while merge master to not get too far behind to not make the merge hard

## Forking

**Description**: Copy repository to your account (cloud)

## Pull (not pull request)

**Decription**: Getting the code of a repository</br>
**Code**: git pull <ins>_link of repo_</ins> </br>

## Notes

1. Every contributer will have its branch
2. Merging happens to only 2 branches
3. Merging has a base branch, where the changes will get committed
4. Merging are pull requests

## Pull Requests

1. Fetch remote branches
2. git checkout main
3. git pull origin main
4. git checkout branchToBeMerged
5. In source control do pull request (not the built in, which is ironically built in)
6. click resolve confilcts
7. when done click commit changes
