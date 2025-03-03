import os
import re
import subprocess
import pandas as pd
from datasets import load_dataset

ds = load_dataset("princeton-nlp/SWE-bench_Verified")
df = pd.DataFrame(ds["test"])

repo_name = df["repo"].value_counts().keys()
repositories = list(df["repo"].value_counts().keys())
repositories_with_commit = df[["repo", "base_commit"]]

######################################################

# repositories = [
#     "django/django",
#     "sympy/sympy",
#     "sphinx-doc/sphinx",
#     "matplotlib/matplotlib",
#     "scikit-learn/scikit-learn",
#     "astropy/astropy",
#     "pydata/xarray",
#     "pytest-dev/pytest",
#     "pylint-dev/pylint",
#     "psf/requests",
#     "mwaskom/seaborn",
#     "pallets/flask"
# ]

# astropy/astropy d16bfe05a744909de4b27f5875fe0d4ed41ce607
# astropy/astropy 298ccb478e6bf092953bca67a3d29dc6c35f6752
# astropy/astropy 6ed769d58d89380ebaa1ef52b300691eefda8928
# astropy/astropy 6500928dc0e57be8f06d1162eacc3ba5e2eff692
# astropy/astropy 19cc80471739bcb67b7e8099246b391c355023ee
# astropy/astropy 0df94ff7097961e92fd7812036a24b145bc13ca8
# astropy/astropy 5250b2442501e6c671c6b380536f1edb352602d1
# astropy/astropy 1a4462d72eb03f30dc83a879b1dd57aac8b2c18b
# astropy/astropy a5917978be39d13cd90b517e1de4e7a539ffaa48
# astropy/astropy cdb66059a2feb44ee49021874605ba90801f9986
# astropy/astropy 7269fa3e33e8d02485a647da91a5a2a60a06af61
# astropy/astropy fa4e8d1cd279acf9b24560813c8652494ccd5922

######################################################


def clone_repository(store_path):
    repositories = [
        "django/django",
        "sympy/sympy",
        "sphinx-doc/sphinx",
        "matplotlib/matplotlib",
        "scikit-learn/scikit-learn",
        "astropy/astropy",
        "pydata/xarray",
        "pytest-dev/pytest",
        "pylint-dev/pylint",
        "psf/requests",
        "mwaskom/seaborn",
        "pallets/flask",
    ]

    # Function to clone a repository
    def clone_repository(repo, destination):
        url = f"https://ghp.ci/https://github.com/{repo}.git"
        try:
            print(f"Cloning repository: {repo} into {destination}")
            subprocess.run(["git", "clone", url, destination], check=True)
            print(f"Successfully cloned: {repo} into {destination}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo}: {e}")

    # Iterate over the repositories and clone them
    for repo in repositories:
        destination_path = os.paht.join(store_path, repo.split("/")[-1])
        clone_repository(repo, destination_path)


def reset_repositories(ds_ori_path, ds_content_path, repositories_with_commit):
    def reset_repository(ori_path, repo_path, repo_name, hash):
        try:
            # Copy the repository to a new location
            print(
                f"Copying repository: {repo_name} from {ori_path}/{repo_name} to {repo_path}/{repo_name}_{hash}"
            )
            subprocess.run(
                [
                    "cp",
                    "-rf",
                    f"{ori_path}/{repo_name}",
                    f"{repo_path}/{repo_name}_{hash}",
                ],
                check=True,
            )

            # Reset the repository to the base commit
            print(f"Resetting repository: {repo_name}_{hash} to hash: {hash}")
            subprocess.run(
                ["git", "checkout", hash],
                cwd=f"{repo_path}/{repo_name}_{hash}",
                check=True,
            )
            subprocess.run(
                ["git", "clean", "-f", "-d"],
                cwd=f"{repo_path}/{repo_name}_{hash}",
                check=True,
            )

            # 删除 .git 目录
            subprocess.run(
                ["rm", "-rf", f"{repo_path}/{repo_name}_{hash}/.git"], check=True
            )
            print(
                f"Successfully reset: {repo_name} to hash: {hash} and removed .git directory."
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to reset {repo_name} to hash {hash}: {e}")

    for index, row in repositories_with_commit.iterrows():
        repo_url = row["repo"]
        repo_name = repo_url.split("/")[-1]
        commit_hash = row["base_commit"]

        # if "sympy" not in repo_name:
        reset_repository(ds_ori_path, ds_content_path, repo_name, commit_hash)

    # # delete original repositories
    # for index, row in repositories_with_commit.iterrows():
    #     repo_url = row["repo"]
    #     repo_name = repo_url.split("/")[-1]
    #     subprocess.run(["rm", "-rf", f"{repo_path}/{repo_name}"], check=True)


def create_dataset(ds_label_path):
    # Extract functions and classes from the diff text
    def extract_functions_and_classes(diff_text):
        func_pattern = re.compile(
            r"\s*def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(.*\):", re.MULTILINE
        )
        class_pattern = re.compile(
            r"\s*class\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(.*\):", re.MULTILINE
        )

        functions = func_pattern.findall(diff_text)
        classes = class_pattern.findall(diff_text)

        return functions, classes

    df["label"] = df["patch"].apply(extract_functions_and_classes)
    df["save_path"] = df.apply(
        lambda row: f"{row['repo'].split('/')[-1]}_{row['base_commit']}", axis=1
    )

    dataset = df[
        [
            "repo",
            "base_commit",
            "patch",
            "problem_statement",
            "hints_text",
            "created_at",
            "version",
            "label",
            "save_path",
        ]
    ]

    dataset.to_csv(os.path.join(ds_label_path, "own_dataset.csv"))
    dataset.to_json(
        os.path.join(ds_label_path, "own_dataset.jsonl"), orient="records", lines=True
    )


if __name__ == "__main__":
    ds_ori_path = "dataset/ds_ori"
    ds_label_path = "dataset/ds_label"
    ds_content_path = "dataset/ds_content"

    # Step 1 - Clone repositories
    print("Cloning Repositories..")
    clone_repository(ds_ori_path)

    # Step 2 - Reset repositories to the base commit
    print("Resetting Repo..")
    reset_repositories(ds_ori_path, ds_content_path, repositories_with_commit)

    # Step 3 - Generate dataset
    print("Generating dataset..")
    create_dataset(ds_label_path)
