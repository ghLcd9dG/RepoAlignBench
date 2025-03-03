# export PATH="$HOME/.local/bin:$PATH"
# ctags -R --languages=Python --fields=+n --extras=+q --exclude=*.test.py --exclude=*.spec.py --exclude=*.pyc --exclude=**/test*/ --exclude=env --tag-relative -f ctags ./

import os
import re
import requests
import subprocess
import pandas as pd
from datasets import load_dataset


def ctags_iter_project(dataset_path):
    subprocess.run(["mkdir", "-p", "ctags"])

    for proj in os.listdir(dataset_path):
        proj_path = os.path.join(dataset_path, proj)
        if os.path.isdir(proj_path):
            subprocess.run(
                [
                    "ctags",
                    "-R",
                    "--languages=Python",
                    "--fields=+ne",
                    "--extras=+q",
                    "--sort=no",
                    "--exclude=*.test.py",
                    "--exclude=*.spec.py",
                    "--exclude=*.pyc",
                    "--exclude=**/test*/",
                    "--exclude=env",
                    "--tag-relative",
                    "--output-format=json",
                    "-f",
                    f"ctags/{proj}.tags",
                    proj_path,
                ]
            )


def extract_keywords_from_statement(statement):
    variables = re.findall(r"\b[a-zA-Z_][a-zA-Z_0-9]*\b", statement)

    types = re.findall(
        r"\b[a-zA-Z_][a-zA-Z_0-9]*\s*:\s*[a-zA-Z_][a-zA-Z_0-9]*\b", statement
    )
    methods = re.findall(r"\bdef\s+([a-zA-Z_][a-zA-Z_0-9]*)\b", statement)
    classes = re.findall(r"\bclass\s+([a-zA-Z_][a-zA-Z_0-9]*)\b", statement)

    keywords = set(variables + types + methods + classes)

    keywords = [kw for kw in keywords if len(kw) > 2]
    with open("resources/stopwords-en.txt", "r") as f:
        stop_words = f.read().splitlines()

    keywords = [kw for kw in keywords if kw.lower() not in stop_words]
    return keywords


def find_matching_lines_with_pandas(jsonl_file, keywords):
    try:
        df = pd.read_json(jsonl_file, lines=True)
    except Exception as e:
        print(f"Error reading JSONL file: {e}")
        return []

    matching_lines = []

    for index, row in df.iterrows():
        try:
            pattern = str(row["pattern"])
        except KeyError:
            print(f"KeyError: 'pattern' column missing at index {index}")
            continue

        if "test" in pattern or "Test" in pattern:
            continue

        keyword_count = sum(1 for kw in keywords if kw in pattern)
        if keyword_count > 0:
            matching_lines.append((dict(row.items()), keyword_count))

    matching_lines.sort(key=lambda x: x[1], reverse=True)

    # 过滤出大于或等于3的行
    matching_lines = [line for line in matching_lines if line[1] >= 3]

    # 输出前10个
    matching_lines = matching_lines[:10]

    return matching_lines


import requests
import time


def query_LLM(ori_query, context, max_retries=3, delay=2):
    prompt = f"""
### Task Description
Imagine you are a helpful assistant for a software developer who is responsible for fixing bugs and implementing new features. Now you need to assist the developer in crafting a query to retrieve relevant functions from a codebase.
Transform GitHub issue details and ctags code context into an optimized prompt for the CodeBERT model to retrieve the most relevant functions. 

Your Output should be a query that guides the model to return the most relevant functions related to the issue.

### Context
Given a GitHub issue describing a bug or feature request, along with ctags-formatted code context containing function signatures, your task is to craft a query that will guide the CodeBERT model to return the most relevant functions related to the issue.

### Issue
{ori_query}

### Ctags Context
{context}

IMPORTANT: ONLY return the rewritten query, with no additional information.
"""
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 4096,
        "temperature": 0.7,
    }
    PROXY_URL = "http://192.168.211.164:5000/api/openai"

    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.post(PROXY_URL, json=data)

            if response.status_code == 200:
                print("Response from OpenAI Proxy:")
                print(response.json())
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Error: {response.status_code}")
                print(response.json())
                break
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

        attempt += 1
        if attempt < max_retries:
            print(f"Retrying... ({attempt}/{max_retries})")
            time.sleep(delay)
        else:
            print("Max retries reached, request failed.")
            break
    return None


def main():
    dataset_path = "/home/liuaofan/code_nlpl/own_csn4/dataset/ds_content"
    ctags_path = "/home/liuaofan/code_nlpl/own_csn4/ctags"

    dataset = pd.read_json("dataset/ds_label/own_dataset.jsonl", lines=True)
    dataset["keywords"] = dataset["problem_statement"].apply(
        extract_keywords_from_statement
    )
    dataset["keyword_num"] = dataset["keywords"].apply(len)

    def process_dataset(dataset):
        for idx, row in dataset.iterrows():
            if row["keyword_num"] > 0:
                keywords = row["keywords"]
                save_path = row["save_path"]

                proj_path = os.path.join(ctags_path, save_path)
                if not proj_path.endswith(".tags"):
                    proj_path += ".tags"

                print(keywords)
                print(proj_path)

                matched_lines = find_matching_lines_with_pandas(proj_path, keywords)
                print(f"Matched lines count: {len(matched_lines)}")

                rewritten_query = query_LLM(row["problem_statement"], matched_lines)
                dataset.loc[idx, "rewritten_query"] = rewritten_query

        return dataset

    dataset = process_dataset(dataset)

    output_path = "dataset/ds_label/own_dataset_rewritten.jsonl"
    dataset.to_json(output_path, orient="records", lines=True)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()
