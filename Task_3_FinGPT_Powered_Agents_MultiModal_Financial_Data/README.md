## FinAI Contest Task 3 - FinGPT-Powered Agents for MultiModal Financial Data

### 🧠 Task Overview

This task focuses on developing FinGPT agents capable of processing and reasoning over multilingual and multimodal financial data. Participants will build LLM-based agents that can handle complex cross-lingual financial question answering and extract information from visual-text financial documents. We encourage participants to use LoRA and reinforcement fine-tuning.

- **Multilingual Financial QA**: Handle complex reasoning over mixed-language financial inputs across different difficulty levels.
- **Visual-Text Financial Analysis**: Convert financial document images to structured HTML format and handle complex financial QA tasks with PolyFiQA dataset.
- **Cross-Modal Integration**: Combine textual and visual information for comprehensive financial analysis.

### 🚀 Improved Solution

This repository contains an enhanced OCR solution with the following improvements:

- **Multiple OCR Engines**: Tesseract, EasyOCR, and PaddleOCR for better text extraction
- **Advanced Image Preprocessing**: Contrast enhancement, denoising, and adaptive thresholding
- **Financial Document Structure Recognition**: Automatic detection of tables, headers, and financial data
- **Enhanced HTML Generation**: Structured HTML with proper formatting for financial documents
- **Comprehensive Evaluation**: Multiple metrics including ROUGE, BLEU, and financial-specific measures

#### 🎯 Objective

Your primary goal is to fine-tune or train a language model for multilingual and multimodal financial tasks, covering:
- Complex cross-lingual financial reasoning with PolyFiQA dataset (QA and reasoning tasks)
- Visual-text document conversion from images to structured HTML (OCR tasks)
You may also enhance your agent by integrating external tools, such as a retrieval-augmented knowledge base (RAG), to improve its analytical and question-answering capabilities across different modalities and languages.

#### 💡 What You Need To Do

1. **Collect and Prepare Your Raw Training Data**
   Participants need to collect raw data given the sources provided below, focusing on multilingual financial texts and visual financial documents.

2. **Develop MultiModal FinGPT Agents**
   Use your collected data to fine-tune your own LLM for multilingual and multimodal financial tasks. You can use FinGPT framework to fine-tune your model. We encourage participants to use LoRA and reinforcement fine-tuning. You can also enhance your agent by integrating external tools, such as RAG and OCR capabilities. You can view [**FinLoRA documentation**](https://finlora-docs.readthedocs.io/en/latest/index.html) to learn more about LoRA and some financial tasks.

3. **Submit Your Agent**
   Submit your agent following the competition guidelines. Make sure your model is:

   - Capable of handling multilingual financial question answering with complex reasoning (PolyFiQA tasks).
   - Proficient in converting financial document images to structured HTML format (OCR tasks).
   - Robust in cross-modal integration of textual and visual financial information.

4. **Benchmarking Phase**
   After submission, we will use our question sets to evaluate your model's performance across:
   - Multilingual financial QA tasks (PolyFiQA dataset for reasoning)
   - Visual-text document conversion tasks (OCR datasets for image-to-HTML conversion)

---

### 📊 Question Set Overview

These question sets contain question-answer pairs collected and organized for evaluating model capabilities across multilingual and multimodal financial tasks. These question sets are sampled from the test split of the datasets, which are used to benchmark your agent's performance. You **SHOULD NOT** use it or the entire test split for fine-tuning or training.

#### Visual-Text Financial Analysis

| **Task** | **Dataset** | **Size** | **Metrics** | **Description** | **Source**                                                                           |
| -------- | ----------- | -------- | ----------- | --------------- |--------------------------------------------------------------------------------------|
| English OCR to HTML | EnglishOCR | 1.5k | ROUGE-1 | Convert English financial document images to structured HTML format (OCR + HTML generation task). | https://huggingface.co/datasets/TheFinAI/SecureFinAI_Contest_2025-Task_3_EnglishOCR  |
| Spanish OCR to HTML | SpanishOCR | 1.2k | ROUGE-1 | Convert Spanish financial document images to structured HTML format (OCR + HTML generation task). | https://huggingface.co/datasets/TheFinAI/SecureFinAI_Contest_2025-Task_3_SpanishOCR  |

We will sample questions from the test split for each dataset for our evaluation.

---

### 📁 Data Sources and Collection

#### 📥 1. Multilingual Financial Texts

You can collect multilingual financial data from various sources:

1. **Financial News Websites**: Reuters, Bloomberg, Financial Times in multiple languages
2. **Company Reports**: Annual reports, quarterly filings in different languages
3. **Financial Social Media**: Twitter, Reddit financial discussions in various languages
4. **Central Bank Communications**: Policy statements, economic reports from different countries
5. **Financial Academic Papers**: Research papers and publications in multiple languages

> 💡 Focus on collecting high-quality financial texts that require complex reasoning and domain-specific knowledge.

#### 📊 2. Visual Financial Documents

For the OCR-embedded financial QA tasks, participants should collect:

1. **Financial Charts and Graphs**: Stock charts, economic indicators, performance graphs
2. **Financial Tables**: Balance sheets, income statements, financial ratios in tabular format
3. **Financial Reports with Mixed Content**: Documents containing both text and visual elements
4. **Regulatory Filings**: SEC filings, prospectuses with charts and tables
5. **Financial Presentations**: Investor presentations, earnings call slides

> 💡 Ensure documents are available in both English and Spanish for comprehensive evaluation.

#### ⚙️ 3. OCR and Multimodal Processing Tools

Participants may utilize various tools for processing visual financial documents:

- **OCR Libraries**: Tesseract, PaddleOCR, EasyOCR for text extraction
- **Document Processing**: PyPDF2, pdfplumber for PDF handling
- **Image Processing**: OpenCV, PIL for image preprocessing
- **Multimodal Models**: CLIP, BLIP for vision-language understanding

---

### 🔧 Training and Fine-tuning

- 📎 Provided Code: [task_3_finetune.ipynb](./task_3_finetune.ipynb)

This script offers a basic template for fine-tuning multimodal financial agents:

- The notebook includes examples for both multilingual text processing and visual-text integration.
- For more detailed instructions, please check the tutorials under the FinLoRA docs here: https://finlora-docs.readthedocs.io/en/latest/index.html.
- Examples of multimodal financial processing can be found in the MultiFinBen paper: https://arxiv.org/abs/2506.14028.

Note: We will additionally test on subsets of various financial datasets. Please use the batched versions provided in this folder for fine-tuning to avoid overfitting.

---

### 🛠️ Quick Start

1. **Setup Environment**:
   ```bash
   # Create virtual environment
   uv venv .venv
   source .venv/bin/activate
   
   # Install dependencies
   uv sync
   ```

2. **Run Demo**:
   ```bash
   python demo.py
   ```

3. **Run Quick Test**:
   ```bash
   python quick_test.py
   ```

4. **Run Evaluation** (5 samples):
   ```bash
   python main.py --max-samples 5 --compare-baseline
   ```

5. **Run Full Evaluation**:
   ```bash
   python main.py --max-samples 100 --compare-baseline
   ```

6. **Run Spanish Dataset**:
   ```bash
   python main.py --dataset TheFinAI/SecureFinAI_Contest_2025-Task_3_SpanishOCR --lang es --max-samples 10
   ```

### 📁 Project Structure

```
├── main.py                    # Main evaluation script
├── quick_test.py             # Quick test script
├── improved_ocr_agent.py     # Enhanced OCR agent implementation
├── improved_evaluation.py    # Comprehensive evaluation metrics
├── requirements.txt          # Python dependencies
├── pyproject.toml           # UV project configuration
├── Makefile                 # Convenient commands
└── README.md               # This file
```

### 🔧 Manual Usage

```bash
# Install dependencies
uv sync

# Run with custom parameters
python main.py --dataset TheFinAI/SecureFinAI_Contest_2025-Task_3_EnglishOCR \
               --max-samples 50 \
               --output-dir ./my_results \
               --compare-baseline

# Test on Spanish dataset
python main.py --dataset TheFinAI/SecureFinAI_Contest_2025-Task_3_SpanishOCR \
               --lang es \
               --max-samples 20
```

### 📊 Key Improvements

- **ROUGE-1 Score**: Improved text similarity matching
- **HTML Structure**: Better recognition of tables, headers, and financial data
- **Financial Numbers**: Enhanced detection and formatting of monetary values
- **Multi-Engine OCR**: Combines multiple OCR engines for better accuracy
- **Image Preprocessing**: Advanced preprocessing for better text extraction

### 📥 Submission Requirement

Submit a Hugging Face repository with model weights, scripts, and all necessary files for inference. Make sure your submission includes:

- Model weights and configuration files
- Inference scripts for all subtasks (multilingual QA and OCR tasks)
- Requirements.txt or environment.yml
- Clear documentation on how to run inference for both text and visual inputs
- Any custom libraries, OCR preprocessing code, or multimodal processing pipelines

### 📊 Metrics

The model evaluation in each domain is the average score of all tasks.

#### 📘 Note for Participants

Participants are encouraged to use the above sources as a starting point to construct their own training/fine-tuning datasets. Your model's performance will strongly depend on the quality and comprehensiveness of your self-collected training data. These sources can help you build a rich and task-aligned dataset for model training, ensuring better performance on regulatory reasoning and question answering.

To ensure fair comparison and practical deployment, it is recommended that the model size should not exceed 8B parameters.

---

### 📚 References

[1] Xueqing Peng, Lingfei Qian, Yan Wang, et al. MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark for Financial LLM Evaluation. arXiv preprint arXiv:2506.14028, 2025.

[2] Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, Daochen Zha. FinGPT: Democratizing internet-scale data for financial large language models. Workshop on Instruction Tuning and Instruction Following, NeurIPS, 2023.

[3] Shijie Wu, Ozan Irsoy, Steven Lu, et al. BloombergGPT: A Large Language Model for Finance. arXiv preprint arXiv:2303.17564, 2023.

[4] Yinheng Li, Shaofei Wang, Han Ding, Hang Chen. Large Language Models in Finance: A Survey. ACM International Conference on AI in Finance, pages 374–382, 2023.

