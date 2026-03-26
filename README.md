# Restaurant Review ABSA System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](YOUR_STREAMLIT_APP_URL)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](YOUR_HUGGINGFACE_MODEL_URL)

ISOM5240 Deep Learning Course Project - Aspect-Based Sentiment Analysis for Restaurant Reviews

## 🎯 Project Overview

This project implements an Aspect-Based Sentiment Analysis (ABSA) system for restaurant reviews using state-of-the-art transformer models from Hugging Face. The system consists of two pipelines:

1. **Pipeline 1 - Aspect Category Detection**: Multi-label classification to identify mentioned aspects (Food, Service, Price, Ambience, Anecdotes/Miscellaneous)
2. **Pipeline 2 - Aspect Sentiment Analysis**: 4-class classification to determine sentiment (Positive, Negative, Neutral, Conflict)

## 🚀 Live Demo

**Streamlit Cloud App**: [YOUR_APP_URL](YOUR_APP_URL)

**Hugging Face Models**:
- Aspect Detection: [YOUR_HF_USERNAME/absa-aspect-detection](YOUR_HF_URL)
- Sentiment Analysis: [YOUR_HF_USERNAME/absa-sentiment-analysis](YOUR_HF_URL)

## 📊 Dataset

- **Source**: SemEval-2014 Task 4
- **Training Set**: 3,044 reviews
- **Validation Set**: 100 reviews
- **Test Set**: 1,600 reviews (Phase A + Phase B)

## 🏗️ Model Architecture

### Pipeline 1: Aspect Detection
- **Base Model**: DistilBERT-base-uncased
- **Task**: Multi-label Classification
- **Labels**: Food, Service, Price, Ambience, Anecdotes/Miscellaneous
- **Evaluation Metrics**: F1-micro, Precision, Recall

### Pipeline 2: Sentiment Analysis
- **Base Model**: RoBERTa-base
- **Task**: Multi-class Classification
- **Labels**: Positive, Negative, Neutral, Conflict
- **Evaluation Metrics**: Accuracy, F1-weighted

## 🛠️ Installation

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/absa-restaurant-reviews.git
cd absa-restaurant-reviews

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Model Files

Place the fine-tuned model files in the following structure:
```
.
├── app.py
├── requirements.txt
├── README.md
├── aspect_detection_model/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── sentiment_analysis_model/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

## 📖 Usage

1. **Enter a restaurant review** in the text area
2. **Click "Analyze Review"** button
3. **View results**:
   - Detected aspects with confidence scores
   - Sentiment analysis for each aspect
   - Visualizations (bar charts, pie charts)
   - Summary table

### Example Reviews

- "The food was delicious but the service was slow."
- "Great atmosphere and friendly staff!"
- "The pasta was overcooked but the dessert was amazing."

## 📁 Project Structure

```
GroupXX_学号/
├── GroupXX_documentation/
│   ├── Project_report.pdf
│   └── Experimental_results.xlsx
├── GroupXX_program/
│   ├── Python_notebooks/
│   │   ├── 1_data_preprocessing.ipynb
│   │   ├── 2_pipeline1_aspect_detection.ipynb
│   │   ├── 3_pipeline2_sentiment_analysis.ipynb
│   │   └── 4_experiments.ipynb
│   └── GitHub_App_Files/
│       ├── app.py
│       ├── requirements.txt
│       └── README.md
├── GroupXX_Dataset_files/
│   ├── train_data.json
│   ├── trial_data.json
│   └── Fine-tuned_Model_files/
└── GroupXX_presentation/
    ├── Presentation_slide.pptx
    └── grpXX.mp4
```

## 📊 Experimental Results

| Model | Pipeline 1 F1 (micro) | Pipeline 2 Accuracy | Training Time |
|-------|----------------------|---------------------|---------------|
| DistilBERT | 0.XXX | 0.XXX | XXs |
| RoBERTa | 0.XXX | 0.XXX | XXs |
| BERT | 0.XXX | 0.XXX | XXs |

See [Experimental_results.xlsx](GroupXX_documentation/Experimental_results.xlsx) for detailed results.

## 🧪 Reproducing the Experiments

All experiments are documented in Jupyter Notebooks:

1. **Data Preprocessing**: `1_data_preprocessing.ipynb`
2. **Aspect Detection Training**: `2_pipeline1_aspect_detection.ipynb`
3. **Sentiment Analysis Training**: `3_pipeline2_sentiment_analysis.ipynb`
4. **Experiments & Comparisons**: `4_experiments.ipynb`

Run these notebooks in Google Colab with GPU enabled for best performance.

## 📝 Citation

```bibtex
@inproceedings{pontiki-etal-2014-semeval,
  title = {SemEval-2014 Task 4: Aspect Based Sentiment Analysis},
  author = {Pontiki, Maria and Galanis, Dimitris and Pavlopoulos, John and Papageorgiou, Haris and Androutsopoulos, Ion and Manandhar, Suresh},
  booktitle = {Proceedings of the 8th International Workshop on Semantic Evaluation},
  year = {2014}
}
```

## 👥 Team Members

- Student Name (Student ID)
- [Add team members if applicable]

## 📄 License

This project is for educational purposes (ISOM5240 Deep Learning Course).

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- SemEval-2014 organizers for the dataset
- Streamlit for the deployment platform
