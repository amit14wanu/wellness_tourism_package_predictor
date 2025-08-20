from huggingface_hub import HfApi, create_repo
import os
from huggingface_hub.utils import RepositoryNotFoundError   


repo_id = "amit14official/tourism-project"
repo_type = "space"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="tourism_project/deployment", # the local folder containing your files
    repo_id=repo_id,                          # the target repo
    repo_type=repo_type,                        # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
    space_sdk="streamlit",                     # optional: specify the SDK for the space
)