# 🖼️ Image Captioning with CNN–LSTM (TensorFlow)

> Generate natural language captions for images using a deep learning pipeline that combines a **CNN encoder** (InceptionV3) and an **LSTM decoder**.  
> This project replicates the structure of modern image-to-text models and demonstrates a full ML workflow — from data preprocessing to model training and evaluation.

---

### 🧠 Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-red?logo=keras)
![NumPy](https://img.shields.io/badge/NumPy-lightblue?logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-blue?logo=plotly)
![Colab](https://img.shields.io/badge/Google%20Colab-Notebook-yellow?logo=googlecolab)

---

## 📘 Project Overview

This repository implements an **image captioning model** trained on the **Flickr8k dataset**.  
It combines **computer vision** and **natural language processing** techniques to describe images in plain English.

### Key Features
- 🧩 Modular architecture (separate data, model, and training scripts)
- 🧠 Encoder–decoder model using CNN (InceptionV3) and LSTM
- 🗂️ Clean code organization (`src/`, `utils/`, `data/`, `outputs/`)
- 📈 Automatic training logs and history visualization
- 🔁 Easy reproducibility on Google Colab

---

## 🧩 Model Architecture

```
Image → CNN Encoder (InceptionV3) → Feature Vector
             ↓
         LSTM Decoder → Caption Tokens → Text Output
```

- **Encoder:** Extracts visual features from images using a pre-trained InceptionV3 CNN  
- **Decoder:** Generates captions word-by-word using LSTM layers  
- **Loss:** Sparse categorical cross-entropy  
- **Optimizer:** Adam  

A diagram of the model is available as `model.png`.

---

## 🚀 How to Run (on Google Colab)

You can reproduce the entire project easily using Google Colab:

```python
# 1️⃣ Clone the repository
!git clone https://github.com/<your-username>/image-captioning-tensorflow.git
%cd image-captioning-tensorflow

# 2️⃣ Install dependencies
!pip install -r requirements.txt

# 3️⃣ Download the Flickr8k dataset (automated)
!python utils/data_downloader.py

# 4️⃣ Run the training notebook
!jupyter nbconvert --to notebook --execute main.ipynb
```

> 💡 *Alternatively, open `main.ipynb` directly in Colab and run all cells.*

---

## 📂 Project Structure

```
image_captioning_project/
│
├── main.ipynb                # End-to-end notebook (data → training → evaluation)
├── model.png                 # Model diagram
├── training_history.pkl/json # Training history and metrics
│
├── src/
│   ├── model.py              # Defines the CNN–LSTM architecture
│   ├── train.py              # Training loop and checkpoint saving
│   ├── evaluate.py           # Caption generation and BLEU scoring
│   ├── feature_extractor.py  # Image feature extraction using InceptionV3
│   ├── data_generator.py     # Custom data generator for large datasets
│   ├── tokenizer_utils.py    # Tokenizer and vocabulary utilities
│   ├── text_processing.py    # Caption cleaning and preprocessing
│   └── data_utils.py         # Data loading and split helpers
│
├── utils/
│   ├── data_downloader.py    # Script to download and unzip Flickr8k dataset
│   └── file_manager.py       # Folder and file handling utilities
│
├── data/
│   ├── captions/             # Caption text files (Flickr8k)
│   └── images/               # Placeholder for image dataset
│
└── outputs/
    └── predictions/          # Model outputs and generated captions
```

---

## 🧪 Evaluation Metrics

- **BLEU score** for caption quality  
- **Training loss** and **validation accuracy** tracked over epochs  
- Visualization of learning curves from `training_history.pkl`  

---

## 📦 Requirements

Create a `requirements.txt` (or install manually):

```
tensorflow>=2.12
numpy>=1.23
tqdm>=4.65
requests>=2.31
```

---

## 💡 Future Improvements

- Integrate **attention mechanisms (Bahdanau / Luong)**
- Train on **Flickr30k** or **MS-COCO** for richer captions
- Deploy inference as a **Flask / Streamlit web app**
- Experiment with **Transformer-based decoders**

---

## 🧑‍💻 Author

**[Your Name]**  
💼 AI / ML Engineer | Deep Learning & Computer Vision  
📫 [Your LinkedIn](https://www.linkedin.com/) • [Your Portfolio](https://github.com/yourusername)

---

## 🏷️ License

This project is licensed under the MIT License — feel free to use and modify for research or learning.

---

### ⭐ If you find this project helpful, consider giving it a star!
