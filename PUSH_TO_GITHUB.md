Goal: push this forecasting project into a subfolder of the remote repo `https://github.com/Homssalomssa/bvmt-sentiment-analysis`.

Recommended safe workflow (manual, easy to audit):

1) Create a branch locally to prepare the change:

```bash
# from project root (this folder)
git checkout -b add-bvmt-forecasting
```

2) Create a subfolder and move project files into it (manual move is safest):

```bash
mkdir bvmt-forecasting
# Move files (example)
# On Windows (PowerShell)
Move-Item -Path * -Destination bvmt-forecasting -Exclude .git,.gitignore,.venv
# On Unix
mv $(ls | grep -v "^.git$\|^requirements.txt$") bvmt-forecasting/
```

Verify the set of files moved and keep any repo-level files you want at root (like CI configs).

3) Commit and push to a new branch on the target repo:

```bash
git add .
git commit -m "Add BVMT forecasting module in subfolder bvmt-forecasting"
# Add remote (only if not already added)
git remote add target https://github.com/Homssalomssa/bvmt-sentiment-analysis.git
git push target HEAD:refs/heads/add-bvmt-forecasting
```

4) Open a Pull Request on GitHub from `add-bvmt-forecasting` -> `main` of the target repo and request merge.

Notes and alternatives:
- If you prefer automation, use `push_subfolder.bat` (Windows) or `push_subfolder.sh` (Linux/macOS). These scripts attempt to move tracked files into a subfolder, commit, add remote and push a branch. Review the script before running.
- Don't run the script if you don't have a clean working tree — stash or commit local changes first.
- If the target repo requires PRs from forks, create a fork and set its URL instead of `target`.

If you'd like, I can: create the branch and run the script for you locally (I can't push to your GitHub without credentials), or prepare a ZIP ready to upload to the repo UI.
