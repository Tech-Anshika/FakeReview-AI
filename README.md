# FakeReview-AI 🕵️‍♂️🚫

**FakeReview-AI** is a robust, hybrid machine learning system designed to detect fraudulent product reviews in E-Commerce environments. It combines state-of-the-art Natural Language Processing (NLP) with behavioral analysis to identify fake reviews with high accuracy.

## 🚀 Features

- **Hybrid Analysis Engine**: Merges **Deep Learning (DistilBERT)** for text content analysis with **Behavioral Heuristics** (user patterns) for a dual-layer detection strategy.
- **Microservices Architecture**: Built with **FastAPI** for high-performance, real-time inference.
- **Deep Learning Model**: Fine-tuned `DistilBERT` model for sentiment and authenticity classification.
- **Behavioral Scoring**: Analyzes metadata such as review frequency, rating deviation, and account age to flag suspicious user activities.
- **REST API**: well-defined endpoints for integration with frontend applications or 3rd party services.

## 🛠️ Tech Stack

- **Language**: Python 3.9+
- **Framework**: FastAPI (Backend API)
- **ML/DL**: Torch, Transformers (Hugging Face), Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Server**: Uvicorn

## 📂 Project Structure

```bash
FakeReview-AI/
├── api/                 # FastAPI application
│   ├── main.py          # API Entry point
│   └── schema.py        # Pydantic models
├── behavior/            # Behavioral analysis logic
├── data/                # Datasets (Raw & Processed)
├── models/              # Trained models (BERT, PKL files)
├── notebooks/           # Jupyter notebooks for EDA and experiments
├── testing/             # Evaluation scripts
├── training/            # Model training scripts
└── requirements.txt     # Python dependencies
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tech-Anshika/FakeReview-AI.git
   cd FakeReview-AI
   ```

2. **Create a Virtual Environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🏎️ Usage

### Running the API Server
To start the fake review detection API locally:

```bash
uvicorn api.main:app --reload
```
The API will be available at `http://127.0.0.1:8000`.

### API Documentation
Once running, visit the interactive Swagger docs at:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

### Prediction Example
Send a `POST` request to `/predict`:

```json
{
  "text": "This product is amazing! I bought it yesterday and it changed my life.",
  "rating": 5,
  "timestamp": "2023-10-27T10:00:00",
  "user_id": "user_123",
  "product_id": "prod_456"
}
```

## 🧠 Model Training

To retrain the models, navigate to the `training/` directory and run the training scripts:
```bash
python training/train_bert.py
python training/train_ml.py
```

## 📊 Evaluation
Run evaluation scripts in `testing/` to verify model performance:
```bash
python testing/evaluate_hybrid_system.py
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request.
