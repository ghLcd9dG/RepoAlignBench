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


def query_model(query, model, indices, tokenizer, top_k=1000):
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
        query_rep = query_rep.mean(dim=1).cpu().numpy()  # 平均池化获得句子表示

    idxs, distances = indices.get_nns_by_vector(
        query_rep.flatten(), top_k, include_distances=True
    )
    return idxs, distances


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
    """
    计算 F1 Score, MRR (Mean Reciprocal Rank), 和 Top-K Accuracy。
    
    参数:
        predictions (list of list): 每个查询的预测结果 URL 列表。
        ground_truth (list of list): 每个查询的真实相关 URL 列表。
        top_k (int): 评估的 Top-K 范围。
    
    返回:
        tuple: 包括 F1 分数, MRR 值, 和 Top-K 准确率。
    """
    f1_scores = []
    mrr_sum = 0
    top_k_correct_count = 0
    valid_query_count = 0

    for query_idx, retrieved_urls in enumerate(predictions):
        true_urls = ground_truth[query_idx]
        
        # 跳过空的预测或真实结果
        if not true_urls or not retrieved_urls:
            continue
        
        true_set = set(true_urls)
        retrieved_set = set(retrieved_urls[:top_k])
        print(len(true_set), len(retrieved_set))
        print(f"true_set: {true_set}")
        print(f"retrieved_set: {retrieved_set}")
        
        # Precision, Recall 和 F1
        true_positive = len(retrieved_set & true_set)
        precision = true_positive / len(retrieved_set) if retrieved_set else 0
        recall = true_positive / len(true_set) if true_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        f1_scores.append(f1)
        
        # MRR
        for rank, url in enumerate(retrieved_urls[:top_k]):
            if url in true_set:
                mrr_sum += 1 / (rank + 1)
                break

        # Top-K Accuracy
        if any(url in true_set for url in retrieved_urls[:top_k]):
            top_k_correct_count += 1

        valid_query_count += 1

    # 平均值计算
    mean_f1 = np.mean(f1_scores) if f1_scores else 0
    mean_mrr = mrr_sum / valid_query_count if valid_query_count > 0 else 0
    top_k_accuracy = top_k_correct_count / valid_query_count if valid_query_count > 0 else 0

    # 返回更精细的小数点结果以帮助调参
    return round(mean_f1, 4), round(mean_mrr, 4), round(top_k_accuracy, 4)


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    df = pd.read_json(
        "dataset/ds_label/own_dataset_label_01.jsonl",
        lines=True,
    )
    definitions = df.to_dict("records")
    queries = list(df["Query"].unique())

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = CodeDataset(definitions, tokenizer)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    code_representations = batch_inference(data_loader, model, device)
    print(f"code_representations shape: {code_representations.shape}")

    indices = AnnoyIndex(code_representations.shape[1], "angular")
    for idx, vector in tqdm(enumerate(code_representations)):
        indices.add_item(idx, vector.numpy())
    indices.build(200)

    predictions = []
    ground_truth = []
    for query in queries:
        nearest_neighbors, distances = query_model(query, model, indices, tokenizer)

        predicted_urls = [definitions[idx]["GitHubUrl"] for idx in nearest_neighbors]
        gt_urls = list(df[(df["Query"] == query) & (df["Relevance"] == 1)]["GitHubUrl"])

        if len(gt_urls) == 0 or len(predicted_urls) == 0:
            continue
        predictions.append(predicted_urls)
        ground_truth.append(gt_urls)

    f1, mrr, top_k_accuracy = compute_metrics(
        predictions, ground_truth, top_k=5
    )
  
    print(f"F1 Score: {f1}")
    print(f"MRR: {mrr}")
    print(f"Top-K Accuracy: {top_k_accuracy}")
