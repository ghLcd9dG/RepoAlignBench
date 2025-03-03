# ReflectCode: Adversarial Reflection-Augmented Code Retrieval Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **Repository-Level Code Retrieval for Change Request-Driven Scenarios**

<div align="center">
  <img src="assets/archv9.png" width="600px" alt="ReflectCode Architecture">
  <p><em>Dual-Tower Architecture with Adversarial Reflection Mechanism</em></p>
</div>

## 📖 Overview
Modern software evolution demands code retrieval systems that understand cross-component change intents. **ReflectCode** addresses this challenge through:

- **Holistic Repository Analysis**: Shift from function-centric to repository-level pattern understanding
- **Adversarial Dual-Tower Architecture**: 
  - 🧠 *Semantic Intent Tower*: LLM-powered cross-module dependency modeling
  - 🧩 *Syntactic Pattern Tower*: Tree-sitter based structural feature extraction
- **Dynamic Fusion Mechanism**: Context-aware integration of syntactic, semantic and dependency features

**Key Achievement**:  
✅ **12.2%** Top-5 Accuracy improvement over SOTA baselines  
✅ First benchmark for change request-driven retrieval (52K test cases)

## 🚀 Quick Start

### Prerequisites
```bash
# Install Tree-sitter dependencies
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    nodejs

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Installation
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/yourusername/reflectcode.git

# Initialize submodules (if already cloned)
git submodule init
git submodule update --init --recursive
```

### Running Pipeline
| Mode       | Command                      | Description                     |
|------------|------------------------------|---------------------------------|
| Training   | `bash scripts/run.sh --mode train` | Start model training            |
| Inference  | `bash scripts/run.sh --mode infer` | Generate code recommendations   |
| Evaluation | `bash scripts/eval.sh`       | Benchmark performance analysis  |

## 📚 Benchmark Details
**CR-Bench** (Change Request Benchmark) characteristics:

| Metric          | Value     |
|-----------------|-----------|
| Total Test Cases| 52,000    |
| Avg. Contexts   | 4.7/file  |
| Languages       | 6         |
| Cross-module    | 68% cases |

RepoAlignBench:
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-RA--Bench-blue)](https://huggingface.co/datasets/yourusername/cr-bench)

RepoAlignBench-Medium:
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-RAMedium--Bench-blue)](https://huggingface.co/datasets/yourusername/cr-bench)

RepoAlignBench-Hard:
[![HF Dataset](https://img.shields.io/badge/🤗%20Dataset-RAHard--Bench-blue)](https://huggingface.co/datasets/yourusername/cr-bench)

## 📦 Submodule Structure
```text
vendor/
├── tree-sitter-python      # Syntax parser for Python
├── tree-sitter-java        # Java AST extraction
├── tree-sitter-php         # PHP language support
└── ...                     # Other language parsers
```

## 📝 Citation
If you use ReflectCode in your research, please cite:

```
Liu, A., Song, S., Li, H., Yang, C., Shu, Z., & Qi, Y. (Year of publication). Beyond Function-Level Search: Repository-Aware Dual-Encoder Code Retrieval with Adversarial Verification. 
```
Bib Format
```bibtex
@article{Liu2024Beyond,
  title={Beyond Function-Level Search: Repository-Aware Dual-Encoder Code Retrieval with Adversarial Verification},
  author={Liu, Aofan and Song, Shiyuan and Li, Haoxuan and Yang, Cehao and Shu, Zishan and Qi, Yiyan},
  journal={arXiv preprint arXiv:2402.14323},
  year={2024}
}
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
