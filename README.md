Sentiment Elicitation from Amazon Reviews
📌 Overview
This project explores sentiment analysis of Amazon food product reviews using machine learning and transformer-based NLP models. By analyzing over 568,000 customer reviews, we investigate patterns in customer sentiment, review helpfulness, temporal trends, and challenges such as class imbalance and neutral sentiment classification.
The project compares RoBERTa, DistilBERT, and a CNN with GloVe embeddings for three-way sentiment classification (positive, negative, neutral).
👨‍💻 Authors
Asfandiyar Safi – 25020221@lums.edu.pk
Shahrez Faisal – 25100235@lums.edu.pk
Emaan Waleed – 26100352@lums.edu.pk
Maryam Rizwan – 26100105@lums.edu.pk
LUMS – Lahore, Pakistan
🔍 Dataset
Source: Amazon Food Reviews dataset
Size: 568,454 reviews
Features: Review text, ratings, helpfulness votes, timestamps
Class Distribution:
Positive (⭐ 4–5) → Majority
Neutral (⭐ 3) → Minority (hardest to classify)
Negative (⭐ 1–2) → Moderate
📊 Exploratory Data Analysis (EDA)
Key insights from dataset analysis:
Reviews are highly skewed toward positive ratings (avg. score: 4.16).
Helpfulness votes are polarized → mostly 0 or fully helpful.
Median review length: ~56 words.
Review volume grows steadily, peaking during seasonal events.
Weak correlation between review length & helpfulness (0.17).
Word cloud and n-gram analysis revealed frequent themes like “great taste”, “highly recommend”, and dietary considerations.
⚙️ Methodology
Preprocessing
Removed punctuation, URLs, special characters, and stopwords.
Standardized text (lowercasing).
Addressed class imbalance via oversampling and weighted loss.
Models
CNN with GloVe embeddings (baseline, local feature extraction).
RoBERTa – Transformer model fine-tuned for classification.
DistilBERT – Lightweight Transformer with high accuracy.
Training Setup
Split: 80/20 stratified train-test.
Optimizers: AdamW (Transformers), Adam (CNN).
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
🚀 Results
Model	Accuracy	Neutral F1	Positive F1	Negative F1	Notes
CNN (GloVe)	40%	0.53	0.92	0.19	Overpredicts neutral
RoBERTa	85.1%	0.45	0.92	0.69	Strong but weak on neutral
DistilBERT	86.7%	0.58	0.94	0.63	Best overall
Key Takeaways
Transformer models (RoBERTa & DistilBERT) outperform CNN.
Neutral sentiment remains the hardest to classify, but oversampling improved detection significantly (+287%).
DistilBERT is the most balanced and efficient model for large-scale sentiment analysis.
💡 Business Implications
Positive/Negative detection (F1 ≥ 0.92) → Strong tool for identifying satisfaction/dissatisfaction.
Neutral detection → Key for identifying uncertain or constructive feedback → helps companies prioritize product improvements.
Potential 15–20% increase in customer satisfaction via targeted responses.
🔮 Future Work
Develop hybrid models combining Transformers with rule-based cues.
Improve interpretability with SHAP and LIME.
Apply models to other Amazon domains (electronics, books, etc.).
Explore advanced architectures (e.g., BART, PEGASUS) for abstractive sentiment summarization.
📚 References
Pang et al. (2002) – Sentiment classification with ML techniques.
Kim (2014) – CNNs for sentence classification.
Devlin et al. (2018) – BERT.
Liu et al. (2019) – RoBERTa.
Sanh et al. (2019) – DistilBERT.
