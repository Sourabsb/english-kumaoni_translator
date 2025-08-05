# English-Kumaoni Translator 🏔️

[[Hugging Face Spaces]](https://huggingface.co/spaces/sourabsb/english-kumaoni_translator)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A sophisticated, hybrid AI model for translating English to Romanized Kumaoni, developed as a comprehensive project exploring fine-tuning and Retrieval-Augmented Generation (RAG).

---

## 🚀 Live Demo

You can try the translator live on Hugging Face Spaces:
**[https://huggingface.co/spaces/sourabsb/english-kumaoni_translator](https://huggingface.co/spaces/sourabsb/english-kumaoni_translator)**

## 📖 About The Project

This project addresses the challenge of creating a translation tool for a low-resource language like Kumaoni. Instead of relying on a single method, it implements a smart, hybrid architecture for optimal performance and accuracy:

* **RAG Cache:** A fast and accurate cache built using Retrieval-Augmented Generation (RAG). It uses a FAISS vector database indexed with over 9,538 English-Kumaoni sentence pairs. If a user's query is similar to an existing entry, it provides the translation instantly.

* **Fine-Tuned Fallback:** For new or unique sentences not found in the cache, the system falls back to a powerful `mBART-large-50` model. This model was fine-tuned using LoRA (Low-Rank Adaptation) on the custom Kumaoni dataset to understand the nuances of the language's grammar and vocabulary.

This entire project involved extensive experimentation with multiple models (`ByT5`, `mT5`) for translation quality and performance. After iterative debugging and evaluation, the final deployment was done using the mBART model.

## ✨ Features

* **Hybrid System:** Combines the speed of a vector database with the generative power of a fine-tuned LLM.
* **High-Quality Translations:** The fine-tuned `mBART` model provides coherent and grammatically aware translations.
* **Interactive UI:** A clean and user-friendly interface built with Gradio.
* **Open Source:** The code, model, and dataset are publicly available.

## 🛠️ Tech Stack

* **Base Model:** `facebook/mbart-large-50-many-to-many-mmt`
* **Fine-Tuning:** PyTorch, Hugging Face `Transformers`, `PEFT` (LoRA)
* **RAG Components:**
    * Framework: `LangChain`
    * Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
    * Vector Store: `FAISS`
* **UI & Deployment:** `Gradio`, `Hugging Face Spaces`

## 📊 Dataset

The model was trained on a unique, custom-built dataset of over 9,538 English to Kumaoni Roman sentence pairs. The dataset was compiled using a hybrid methodology to ensure both breadth and quality:

* **Web Scraping (~40%):** Data was initially sourced from public online Kumaoni resources using custom-built web scraping scripts. All scraped content underwent multiple rounds of manual cleaning, de-duplication, and linguistic refinement to ensure high quality and consistency.
* **Manual Curation (~60%):** The majority of the dataset was created, translated, and verified manually to ensure high linguistic accuracy and natural phrasing.

The dataset is publicly available at:

* **Kaggle:** https://www.kaggle.com/datasets/sourabsinghbora/english-kumaoni-translation-dataset/data
* **Hugging Face:** https://huggingface.co/datasets/sourabsb/english-kumaoni_translation_dataset


## ⚙️ Setup and Run Locally

To run this project on your own machine (requires a GPU):

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Sourabsb/english-kumaoni_translator.gitSourabsb/
    cd english-kumaoni_translator
    ```
2.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```sh
    python app.py
    ```

---

## ⚠️ Project Status

* This model serves as a strong proof-of-concept and a functional baseline for Kumaoni translation. It was trained on a custom dataset of over 9,538 sentence pairs. Consequently, while it handles a wide variety of inputs, its knowledge is limited to the patterns found in this data.

* For sentences with highly specific vocabulary or complex grammar, the translation quality may vary. The most significant path to improving this model's accuracy and robustness is to expand the training dataset into the tens or hundreds of thousands of high-quality examples.

---

Built by **Sourab**. This project showcases a complete end-to-end workflow for developing and deploying a specialized language model.

