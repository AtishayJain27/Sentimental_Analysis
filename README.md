# Sentiment Analysis for Student Reviews on Coursera

## Project Overview
This project performs sentiment analysis on Coursera student reviews using three different NLP models:
1. **VADER** (Valence Aware Dictionary and sEntiment Reasoner)
2. **RoBERTa** (Robustly Optimized BERT Approach)
3. **Transformers Pipeline** (Hugging Face's default sentiment analysis model)

The analysis aims to understand student sentiment towards various Coursera courses by examining over 1.4 million reviews.

## Dataset
The dataset used is "Course Reviews on Coursera" from Kaggle, containing:
- 1.45 million course reviews
- Review text, reviewer names, dates, ratings (1-5), and course IDs
- 622 unique Coursera courses

Dataset link: [Course Reviews on Coursera](https://www.kaggle.com/datasets/siddharthm1698/coursera-course-dataset)

## Models Used

### 1. VADER
- Rule-based sentiment analysis tool from NLTK
- Particularly good at handling social media text and short sentences
- Returns polarity scores (negative, neutral, positive) and a compound score

### 2. RoBERTa
- Pretrained transformer model fine-tuned on 124M tweets (2018-2021)
- More sophisticated than VADER, understands context better
- Returns probability scores for negative, neutral, and positive sentiment

### 3. Transformers Pipeline
- Uses Hugging Face's default sentiment analysis model (distilbert-base-uncased-finetuned-sst-2-english)
- Simple API for quick sentiment classification
- Returns label (POSITIVE/NEGATIVE) and confidence score

## Key Findings
1. Rating distribution shows most reviews are positive:
   - 78.79% 5-star ratings
   - 15.58% 4-star ratings
   - Only ~2.3% 1-2 star ratings

2. Course-specific analysis reveals:
   - "what-is-datascience" has the lowest average rating (2.6)
   - "google-cbrs-cpi-training" has the highest average rating (4.93)

3. Sentiment analysis shows:
   - Higher ratings generally correlate with more positive sentiment
   - Some interesting mismatches where positive text had low ratings and vice versa

## Usage

### Requirements
- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, nltk, transformers, scipy

### Installation
```bash
pip install pandas numpy matplotlib seaborn nltk transformers scipy
```

### Running the Analysis
1. Download the dataset from Kaggle
2. Place the CSV file in the correct directory
3. Run the Jupyter notebook or Python script containing the analysis code

## Results Visualization
The project includes several visualizations:
- Rating distribution charts (bar, pie)
- Course-specific rating analysis
- Sentiment score comparisons across models
- Model performance comparisons

## Limitations
1. Dataset size was reduced for processing speed (from 1.4M to 500 samples)
2. Some models had issues with very long reviews (>512 tokens)
3. Sentiment analysis models may misinterpret sarcasm or complex language

## Future Work
- Expand analysis to full dataset with more computing power
- Try additional sentiment analysis models
- Incorporate course metadata for deeper insights
- Build a predictive model for review ratings

## Author
[Your Name]  
[Your Contact Information]  
[Date]
