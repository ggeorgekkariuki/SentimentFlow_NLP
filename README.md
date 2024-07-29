# SentimentFlow: Twitter Sentiment Analysis

<div style="text-align: center;">
  <img src="images\Sentiment_flow_image.jpg" alt="Google vs Apple" title="Google vs Apple" width="600" height="300"/>
</div>

**View this project on Streamlit [here](https://sentimentflow-nlp-project.streamlit.app/)**

## Overview

SentimentFlow addresses the challenge of understanding public sentiment toward products on Twitter. Stakeholders, including companies, marketing teams, and decision-makers, seek insights to make informed strategic decisions based on social media sentiment.

## Problem Statement

The goal is to accurately classify tweet sentiments as positive, negative, or neutral. This classification helps companies gauge customer satisfaction and tailor responses accordingly.

## Stakeholders

1. **Companies (Apple and Google):** Monitor product perception and identify areas for improvement.
2. **Marketing Teams:** Adjust campaigns based on sentiment analysis.
3. **Decision-Makers:** Use insights for product development and brand reputation.

## Value Proposition

Accurate sentiment classification provides actionable insights:
- Negative sentiment: Prompt issue resolution.
- Positive sentiment: Reinforce successful strategies.
- Neutral sentiment: Contextual understanding.

## Objectives

**Main Objective:**
Develop an NLP multiclass classification model with:
- Recall score ≥ 85%
- Accuracy ≥ 80%
- Three sentiment classes: Positive, Negative, and Neutral.

**Specific Objectives:**
1. Identify common words using Word Cloud.
2. Confirm positive and negative word associations.
3. Recognize mentioned products.
4. Analyze sentiment distribution.

## **Data Analysis**

***1. Distribution of the `emotion` feature***

```python
eda.plot_bar(df, feature='emotion', plot='bar')
```


    
![png](images\emotion_distribution.png)

It is evident that class imbalance exists in the data.

```python
eda.plot_fdist()
```

***Frequency Distribution of words used in tweets***
    
![png](images\freq_dist_all_words.png)
    
***Top 10 Bigrams***


```python
eda.bigram_plots(items=10)
```


    
![png](images\bigram.png)
    

***Word Cloud Visual***

Shows the most frequent words used in tweets; those with the highest frequency are larger in size.
```python
eda.create_word_cloud(lemmatized_tweet=df['lemmatized_tweet'])
```


    
![png](images\word_cloud.png)
    


## **Data Modeling**

### ***Models***

The machine learning algorithms used in this section are:

- RandomForest
- Naive Bayes(MultinomialNB)
- LogisticRegression
- DecisionTrees

> The data was vectorized using both `CountVectorizer` and `TfidfVectorizer` to see which vectorizer performed better.

> The best performing models were `RandomForestClassifier` & `LogisticRegression`

### ***1. Random Forest Classifier***

*Before tuning*
```python
modelling(model=rf, cv=True)
modelling(model=rf, tf=True)
```

    Count Vectorisation Results
    
    Main Metrics
    ------------
    Accuracy Score 0.499
    Recall Score 0.502
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.91      0.18      0.30      1541
               1       0.74      0.45      0.56      1523
               2       0.39      0.88      0.54      1500
    
        accuracy                           0.50      4564
       macro avg       0.68      0.50      0.47      4564
    weighted avg       0.68      0.50      0.47      4564
    
    ---------
    TFIDF Vectorisation Results
    
    Main Metrics
    ------------
    Accuracy Score 0.71
    Recall Score 0.71
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.98      0.81      0.89      1541
               1       0.74      0.44      0.55      1523
               2       0.56      0.89      0.68      1500
    
        accuracy                           0.71      4564
       macro avg       0.76      0.71      0.71      4564
    weighted avg       0.76      0.71      0.71      4564


*After hyper-parameter tuning*

```python
# Define the Random Forest classifier
rf = RandomForestClassifier(random_state= 42)

# Define the parameter grid with the necessary hyperparameters
rf_param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30]   # Maximum depth of the tree
}

tuned_rf_cv_model, tuned_rf_tf_model = hyper_tuning(model=rf, params=rf_param_grid, model_name="Random Forest")
```

    Count Vectorisation Results
    
    Best Random Forest Model (Count Vectorization):
     RandomForestClassifier(n_estimators=200, random_state=42)
    
    Test Accuracy (Count Vectorization): 0.704
    
    Test Recall (Count Vectorization): 0.703
    ---------
    
    
    TFIDF Vectorisation Results
    
    Best Random Forest Model (TFIDF Vectorization):
     RandomForestClassifier(n_estimators=200, random_state=42)
    
    Test Accuracy (TFIDF Vectorization): 0.837
    
    Test Recall (TFIDF Vectorization): 0.836
    

#### *Observation*

> The significant improvement in test accuracy from 0.50 to 0.70 in the model using Count Vectorization.

> Note the improvement from 0.71 to 0.837 for the model using TF-IDF Vectorization

> We can note an indication that TF-IDF provides a superior feature representation for the Random Forest model.

### ***2. Logistic Regression***

*Before tuning*
```python
modelling(model=lr, cv=True)
modelling(model=lr, tf=True)
```

    Count Vectorisation Results
    
    Main Metrics
    ------------
    Accuracy Score 0.696
    Recall Score 0.695
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.73      0.91      0.81      1541
               1       0.71      0.65      0.68      1523
               2       0.64      0.53      0.58      1500
    
        accuracy                           0.70      4564
       macro avg       0.69      0.69      0.69      4564
    weighted avg       0.69      0.70      0.69      4564
    
    ---------
    TFIDF Vectorisation Results
    
    Main Metrics
    ------------
    Accuracy Score 0.808
    Recall Score 0.807
    
    Classification Report
                  precision    recall  f1-score   support
    
               0       0.91      0.97      0.94      1541
               1       0.73      0.74      0.74      1523
               2       0.77      0.71      0.74      1500
    
        accuracy                           0.81      4564
       macro avg       0.80      0.81      0.80      4564
    weighted avg       0.81      0.81      0.81      4564

*After hyper-parameter tuning*

```python
# Parameter Tuning
c_space = np.linspace(30, 32, 3)
max_iters = [100, 150, 200]
solvers = ["lbfgs", "liblinear"]
lr_param_grid = { 'C': c_space, 'max_iter':max_iters }
tuned_lr_cv_model, tuned_lr_tf_model = hyper_tuning(model=lr, params=lr_param_grid, model_name="Logistic Regression")
```

    Count Vectorisation Results
    
    Best Logistic Regression Model (Count Vectorization):
     LogisticRegression(C=30.0)
    
    Test Accuracy (Count Vectorization): 0.704
    
    Test Recall (Count Vectorization): 0.702
    ---------
    
    
    TFIDF Vectorisation Results
    
    Best Logistic Regression Model (TFIDF Vectorization):
     LogisticRegression(C=31.0, max_iter=200)
    
    Test Accuracy (TFIDF Vectorization): 0.835
    
    Test Recall (TFIDF Vectorization): 0.834
    

#### *Observation*

> The significant improvement in test accuracy from 0.70 to 0.83 in the Count Vectorization based model.

> TF-IDF Vectorization-based model improved from 0.808 to 0.831.

> Further indication that the TFIDF vectorisation is better