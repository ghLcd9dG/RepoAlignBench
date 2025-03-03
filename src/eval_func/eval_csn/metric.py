import re
import sys
import torch
import wandb
import shutil
import pickle
import pandas as pd

from tqdm import tqdm
from docopt import docopt
from annoy import AnnoyIndex
from wandb.apis import InternalApi
from dpu_utils.utils import RichPath
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from models import CodeDocSimilarityModel
from data_extraction.python.parse_python_data import tokenize_docstring_from_string

def query_model(query, model, indices, language, top_k=1000):
    # Tokenize the query and get its representation
    query_inputs = tokenizer(query, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    query_input_ids = query_inputs['input_ids'].to(device)
    query_attention_mask = query_inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        if torch.cuda.device_count() > 1:
            query_rep = model.module.get_code_representation(query_input_ids, query_attention_mask).cpu().numpy()
        else:
            query_rep = model.get_code_representation(query_input_ids, query_attention_mask).cpu().numpy()
    
    # Search in the Annoy index
    idxs, distances = indices.get_nns_by_vector(query_rep.flatten(), top_k, include_distances=True)
    return idxs, distances

# 定义 Dataset 类，用于批量加载数据
class CodeDataset(Dataset):
    def __init__(self, definitions, tokenizer, max_length=512):
        self.definitions = definitions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.definitions)

    def __getitem__(self, idx):
        code_tokens = self.definitions[idx]["function_tokens"]
        code_tokens = " ".join(code_tokens)
        code_inputs = self.tokenizer(
            code_tokens,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return code_inputs

def batch_inference(data_loader, model, device):
    model.eval()
    representations = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batch"):
            # original 16,1,512
            # input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            # resize to 16,512
            input_ids = batch['input_ids'].squeeze(1).to(device)  # Remove the second dimension (1)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)  # Remove the second dimension (1)
            # print(f"print(input_ids.shape, attention_mask.shape): {input_ids.shape}, {attention_mask.shape}")
            if torch.cuda.device_count() > 1:
                code_representations = model.module.get_code_representation(input_ids, attention_mask)
            else:
                code_representations = model.get_code_representation(input_ids, attention_mask)
            # print(f"code_representations: {code_representations.shape}")
            representations.append(code_representations.cpu())
    return torch.cat(representations, dim=0)


if __name__ == "__main__":
    queries = pd.read_csv("./resources/queries.csv")
    queries = list(queries["query"].values)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = torch.load("./models/train_ln_model_20241105_145820.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
    model.to(device)
    model.eval()
    
    predictions = []
    for language in ("python", "go", "javascript", "java", "php", "ruby"):
    # for language in ["python"]:
        print(f"Evaluating language: {language}")
        pickle_path = f"/home/liuaofan/code_nlpl/code-search-net/data_unzip/{language}/{language}_dedupe_definitions_v2.pkl"
        definitions = pickle.load(open(pickle_path, "rb"))

        # 使用 CodeDataset 和 DataLoader 进行批量处理
        dataset = CodeDataset(definitions, tokenizer)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

        print(f"Extracting code representations for {language}...")
        code_representations = batch_inference(data_loader, model, device)
        print(f"cat code_representations: {code_representations.shape}")

        # 构建 Annoy 索引
        print(f"Building Annoy index for {language}...")
        indices = AnnoyIndex(code_representations.shape[1], "angular")
        for idx, vector in tqdm(enumerate(code_representations)):
            if vector is not None:
                indices.add_item(idx, vector.cpu().numpy())
        indices.build(200)

        # 查询部分
        print(f"Performing queries for {language}...")
        for query in queries:
            nearest_neighbors, query_rep = query_model(query, model, indices, language)
            for idx in nearest_neighbors:
                predictions.append((query, language, definitions[idx]["identifier"], definitions[idx]["url"]))

    # 保存预测结果到 CSV
    df = pd.DataFrame(predictions, columns=["query", "language", "identifier", "url"])
    predictions_csv = "./predictions.csv"
    df.to_csv(predictions_csv, index=False)
