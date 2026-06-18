# 🎵 CD & Vinyl Recommendation System
 
A Two-Tower neural recommender trained on Amazon 5-core CDs & Vinyl (2023 dataset).  
Given a user ID, the system recommends top-10 CDs/vinyls based on purchase history and item features (genre, artist, price, etc.).
You may train and use the model locally, or navigate to our website: https://huggingface.co/spaces/wes1612/CD_Vinyl_Recommendation.
---
 
## 📦 What's Included
 
- **Pre-trained model** (`models/tt_model.pth`) – ready to use
- **Gradio web interface** (`app.py`) – interactive recommendations & similarity search
- **Data files** – subset of Amazon reviews and metadata (already included)
---
 
## 🚀 Quick Start
 
### 1. Install Python (if not already)
 
Requires Python **3.8 – 3.11**.  
Download from [python.org](https://www.python.org/downloads/).
 
### 2. Install dependencies
 
Open a terminal (Command Prompt / PowerShell) inside the project folder and run:
 
```bash
pip install -r requirements.txt
```
 
> 💡 It's recommended to use a virtual environment:
> ```bash
> python -m venv venv
> ```
> Then activate it:
> - **Windows:** `venv\Scripts\activate`
> - **Mac/Linux:** `source venv/bin/activate`
 
### 3. Run the application
 
```bash
python app.py
```
 
Your default browser will open automatically at `http://127.0.0.1:7860`.
 
---
 
## 🧪 What You Can Do
 
- **User Recommendations** – Enter any Amazon user ID from the dataset to get personalised top-10 recommendations (with album covers, titles, and categories).
- **Similar Items** – Enter an item ASIN (e.g., `B0BFRK1FZR`) to find the most similar products based on item embedding cosine similarity.
- **Model Performance** – View evaluation charts comparing BPR vs Two-Tower, and the gap between the 100-negative protocol vs full-ranking.
---
 
## 📂 Project Structure
 
```
.
├── app.py                 # Gradio web interface
├── models/                # Pre-trained model, encoders, item features
│   ├── tt_model.pth
│   ├── user_enc.pkl
│   ├── item_enc.pkl
│   ├── train_binary.pkl
│   ├── item_feat_tensor.pt
│   └── feature_dim.txt
├── data/                  # Amazon raw data (reviews + metadata)
│   ├── CDs_and_Vinyl.csv.gz
│   └── meta_CDs_and_Vinyl.jsonl
├── src/                   # Model definition
│   └── model_twotower.py
├── requirements.txt       # Python dependencies
└── README.md              # This file
```
 
---
 
## ⚠️ Troubleshooting
 
| Issue | Solution |
|---|---|
| `ModuleNotFoundError: No module named 'gradio'` | Run `pip install -r requirements.txt` again. |
| `FileNotFoundError: .../models/tt_model.pth` | Make sure you are running `app.py` from the project root directory (where the `models/` folder is located). |
| Port 7860 already in use | Change the port by editing the last line of `app.py` to `demo.launch(inbrowser=True, server_port=7861)`. |
| Slow first launch | Gradio downloads frontend assets once — this is normal. |
 
---
 
## 📊 Performance Highlights
 
| Metric | Value |
|---|---|
| Two-Tower HR@10 (100-neg protocol) | **0.6227** |
| Two-Tower HR@100 (full ranking) | **0.104** |
| Random baseline HR@100 | ~0.0011 |
 
> The model is **93× better than random** under full-ranking evaluation.
 
---
 
## 🙏 Credits
 
- **Dataset:** [Amazon Review Data (2023)](https://amazon-reviews-2023.github.io/) – Jianmo Ni et al.
- Built with **PyTorch**, **Gradio**, and **scikit-learn**.
Enjoy exploring CD & vinyl recommendations! 🎶
