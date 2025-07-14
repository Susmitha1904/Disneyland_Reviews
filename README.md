
# 🏰 Disneyland Reviews –  Text Summarization & Sentiment Analysis 

This NLP project analyzes visitor reviews from Disneyland parks to understand customer sentiment and generate concise summaries of guest experiences. It applies a complete natural language processing pipeline, including sentiment classification, keyword extraction, and text summarization.

---

## 🎯 Project Objectives

- Classify reviews as **positive**, **negative**, or **neutral** using sentiment analysis
- Generate short summaries from lengthy user reviews using NLP techniques
- Compare sentiment trends across Disneyland locations (California, Paris, Tokyo)
- Visualize frequently mentioned terms and guest feedback themes

---

## 🧠 What I Did

### 🧹 **Text Preprocessing**
- Cleaned review text using tokenization, stopword removal, lemmatization
- Normalized and standardized text for analysis

### 📝 **Text Summarization**
- Applied **frequency-based extractive summarization** to create concise overviews
- Generated top sentence summaries representing key ideas in long reviews
- Compared guest experiences by summarizing positive vs. negative feedback

### 💬 **Sentiment Analysis**
- Used **TextBlob** and **VADER** for rule-based sentiment scoring
- Labeled reviews and analyzed overall sentiment distribution by location
- Visualized polarity scores and trends using bar and pie charts

### 🤖 **Machine Learning Classification**
- Used:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
- Compared models using Accuracy, Precision, Recall, and Confusion Matrix
- **SVM** performed best for clean review classification

---

## 📈 Visualizations

- Word Clouds of most frequent terms (positive vs. negative)
- Bar charts for sentiment distribution by location
- Histograms for review length and polarity scores
- Side-by-side summaries of user reviews before and after compression

---

## 🛠️ Tools & Libraries

| Category         | Tools / Libraries                     |
|------------------|----------------------------------------|
| Programming      | Python                                 |
| NLP Libraries    | NLTK, TextBlob, VADER, spaCy, re        |
| ML Models        | scikit-learn (SVM, NB, Logistic Reg.)   |
| Summarization    | Frequency-based extractive method       |
| Data Handling    | pandas, NumPy                          |
| Visualization    | matplotlib, seaborn, wordcloud         |
| Environment      | Google colab pro                       |

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Susmitha1904/disneyland-reviews-nlp.git
   cd disneyland-reviews-nlp
````

2. Install dependencies:

   ```bash
   pip install pandas numpy nltk matplotlib seaborn wordcloud textblob vaderSentiment scikit-learn
   ```

3. Run the notebook:
---

## 📊 Sample Insights

* **California** park received the highest number of positive reviews
* **Common Complaints**: long wait times, cost, staff behavior
* **Positive Themes**: magical experience, friendly staff, parades
* **Summaries** effectively condensed reviews into 1–2 sentences capturing the review tone

