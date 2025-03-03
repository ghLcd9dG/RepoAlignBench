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
from datasets import load_dataset

from models import CodeDocSimilarityModel

from data_extraction.python.parse_python_data import tokenize_docstring_from_string
from eval_csn.metric import query_model, CodeDataset, batch_inference


# Function to query the model
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

# Dataset class for loading data in batches
class CodeDataset(Dataset):
    def __init__(self, definitions, tokenizer, max_length=512):
        self.definitions = definitions
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.definitions)

    def __getitem__(self, idx):
        code_tokens = self.definitions[idx]["Code"]
        code_tokens = " ".join(code_tokens.split())
        code_inputs = self.tokenizer(
            code_tokens,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": code_inputs["input_ids"].squeeze(0),
            "attention_mask": code_inputs["attention_mask"].squeeze(0),
        }

# Function for batch inference
def batch_inference(data_loader, model, device):
    model.eval()
    representations = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Processing batch"):
            # Remove the second dimension (1)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            if torch.cuda.device_count() > 1:
                code_representations = model.module.get_code_representation(input_ids, attention_mask)
            else:
                code_representations = model.get_code_representation(input_ids, attention_mask)
            representations.append(code_representations.cpu())
    return torch.cat(representations, dim=0)

if __name__ == "__main__":
    # Load dataset and queries
    ds = load_dataset("princeton-nlp/SWE-bench_Verified")
    df = pd.DataFrame(ds["test"])
    queries = list(df["problem_statement"].values)

    # Initialize tokenizer and load model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = torch.load("resources/saved_models/train_ln_model_20241115.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for inference.")
    model.to(device)
    model.eval()
    
    # Load inference data
    infer_data = pd.read_csv('/home/liuaofan/code_nlpl/own_csn2/resources/annotationStore_own_ds_infer.csv', sep=',')
    definitions = infer_data.to_dict('records')

    # Use CodeDataset and DataLoader for batch processing
    dataset = CodeDataset(definitions, tokenizer)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Extract code representations
    code_representations = batch_inference(data_loader, model, device)
    print(f"cat code_representations: {code_representations.shape}")

    # Build Annoy index
    indices = AnnoyIndex(code_representations.shape[1], "angular")
    for idx, vector in tqdm(enumerate(code_representations)):
        if vector is not None:
            indices.add_item(idx, vector.cpu().numpy())
    indices.build(200)

    # Perform queries
    predictions = []
    language = "Python"
    for query in queries:
        nearest_neighbors, query_rep = query_model(query, model, indices, language)
        for idx in nearest_neighbors:
            predictions.append((query, language, definitions[idx]["GitHubUrl"]))

    # Save predictions to CSV
    df = pd.DataFrame(predictions, columns=["query", "language", "url"])
    predictions_csv = "./predictions.csv"
    df.to_csv(predictions_csv, index=False)
