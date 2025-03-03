import os
import re
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from annoy import AnnoyIndex
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset


def query_model_with_annoy(query, model, tokenizer, definitions, top_k=1000):
    indices = build_annoy_index_for_query(query, model, tokenizer, definitions, top_k)

    query_inputs = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    query_input_ids = query_inputs["input_ids"].to(device)
    query_attention_mask = query_inputs["attention_mask"].to(device)

    with torch.no_grad():
        query_rep = model(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        ).last_hidden_state
        query_rep = query_rep.mean(dim=1).cpu().numpy()

    idxs, distances = indices.get_nns_by_vector(
        query_rep.flatten(), top_k, include_distances=True
    )
    return idxs, distances


def build_annoy_index_for_query(query, model, tokenizer, definitions, top_k=1000):
    query_inputs = tokenizer(
        query,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    query_input_ids = query_inputs["input_ids"].to(device)
    query_attention_mask = query_inputs["attention_mask"].to(device)

    with torch.no_grad():
        query_rep = model(
            input_ids=query_input_ids, attention_mask=query_attention_mask
        ).last_hidden_state
        query_rep = query_rep.mean(dim=1).cpu().numpy()

    indices = AnnoyIndex(query_rep.shape[1], "angular")

    for idx, definition in enumerate(definitions):
        if definition["Query"] == query:
            code = definition["Code"]
            code_inputs = tokenizer(
                code,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            code_input_ids = code_inputs["input_ids"].to(device)
            code_attention_mask = code_inputs["attention_mask"].to(device)

            with torch.no_grad():
                code_rep = model(
                    input_ids=code_input_ids, attention_mask=code_attention_mask
                ).last_hidden_state
                code_rep = code_rep.mean(dim=1).cpu().numpy()

            indices.add_item(idx, code_rep.flatten())
    indices.build(200)

    return indices


class CodeDataset(Dataset):
    def __init__(self, definitions, tokenizer, max_length=512):
        self.definitions = definitions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.definitions)

    def __getitem__(self, idx):
        code = self.definitions[idx]["Code"]
        code_inputs = self.tokenizer(
            code,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": code_inputs["input_ids"].squeeze(0),
            "attention_mask": code_inputs["attention_mask"].squeeze(0),
        }


def batch_inference(data_loader, model, device):
    model.eval()
    representations = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            code_representations = outputs.last_hidden_state.mean(dim=1)
            representations.append(code_representations.cpu())

    return torch.cat(representations, dim=0)


def compute_metrics(predictions, ground_truth, top_k):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    mrr_sum = 0
    top_k_correct_count = 0
    valid_query_count = 0

    for query_idx, retrieved_urls in enumerate(predictions):
        true_urls = ground_truth[query_idx]

        if not true_urls or not retrieved_urls:
            continue

        true_set = set(true_urls)
        retrieved_set = set(retrieved_urls[:top_k])

        true_positive = len(retrieved_set & true_set)
        precision = true_positive / len(retrieved_set) if retrieved_set else 0
        recall = true_positive / len(true_set) if true_set else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0
            else 0
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        for rank, url in enumerate(retrieved_urls[:top_k]):
            if url in true_set:
                mrr_sum += 1 / (rank + 1)
                break

        if any(url in true_set for url in retrieved_urls[:top_k]):
            top_k_correct_count += 1

        valid_query_count += 1

    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    mean_precision = np.mean(precision_scores) if precision_scores else 0
    mean_recall = np.mean(recall_scores) if recall_scores else 0
    mean_mrr = mrr_sum / valid_query_count if valid_query_count > 0 else 0
    top_k_accuracy = (
        top_k_correct_count / valid_query_count if valid_query_count > 0 else 0
    )

    return (
        round(mean_f1, 4),
        round(mean_precision, 4),
        round(mean_recall, 4),
        round(mean_mrr, 4),
        round(top_k_accuracy, 4),
    )


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    df = pd.read_json(
        "dataset/ds_label/own_dataset_label_01.jsonl",
        lines=True,
    )
    definitions = df.to_dict("records")
    queries = list(df["Query"].unique())

    models_info = [
        ("microsoft/codebert-base", "codeBERT"),
        # ("Salesforce/codeT5-small", "codeT5"),
        # ("facebook/incoder-1B", "incoder"),
        # ("bigcode/santaCoder", "santaCoder"),
    ]

    for model_name, model_type in models_info:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        predictions = []
        ground_truth = []

        for query in queries:
            nearest_neighbors, distances = query_model_with_annoy(
                query, model, tokenizer, definitions
            )

            predicted_urls = [definitions[idx]["GitHubUrl"] for idx in nearest_neighbors]
            gt_urls = list(df[(df["Query"] == query) & (df["Relevance"] == 1)]["GitHubUrl"])

            if len(gt_urls) == 0 or len(predicted_urls) == 0:
                continue
            predictions.append(predicted_urls)
            ground_truth.append(gt_urls)

        f1, precision, recall, mrr, top_k_accuracy = compute_metrics(
            predictions, ground_truth, top_k=5
        )

        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"MRR: {mrr}")
        print(f"Top-K Accuracy: {top_k_accuracy}")
        print(f"-*-" * 10)
