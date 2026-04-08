from huggingface_hub import HfApi, CommitOperationDelete
import time

api = HfApi()
REPO_ID = "khaiwal009/agri-alert-rl"

# List all files in the repo
all_items = list(api.list_repo_tree(REPO_ID, repo_type="space", recursive=True))
all_files = [
    f.path for f in all_items
    if hasattr(f, 'path') and not f.path.endswith('/')
]

# Files to delete
to_delete = []
for f in all_files:
    if (f.startswith('.venv/') or
        f.startswith('.kiro/') or
        (f.endswith('.md') and f != 'README.md') or
        f == 'validate-submission.sh' or
        f == 'upload_to_hf.py'):
        to_delete.append(f)

print(f"Will delete {len(to_delete)} files in a single commit...")

# Delete all in ONE commit to avoid rate limit
operations = [CommitOperationDelete(path_in_repo=f) for f in to_delete]

api.create_commit(
    repo_id=REPO_ID,
    repo_type="space",
    operations=operations,
    commit_message="chore: remove .venv, .kiro, and spec .md files",
)

print("Done! All unwanted files removed in a single commit.")
