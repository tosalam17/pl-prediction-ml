# Premier League Match Outcome Prediction (Machine Learning)

This project applies **machine learning to sports analytics**, focusing on predicting English Premier League match outcomes using historical match and team performance data. It demonstrates a complete ML workflow—from data preprocessing and feature engineering to baseline modeling and evaluation—on a real-world, noisy classification problem.

The repository is structured and documented to reflect **industry-style ML development practices**, making it suitable for machine learning and data science internship review.

---

## Problem Statement

Predicting football match outcomes is a challenging supervised learning task due to:

* High variance and stochastic outcomes
* Temporal dependencies such as form and momentum
* Contextual effects like home advantage and opponent strength

This project investigates how well **classical machine learning models** can capture these dynamics using engineered team-level features.

---

## Repository Structure

```
pl-prediction-ml/
│
├── data/            # Raw and processed match data
├── notebooks/       # EDA, modeling, and evaluation notebooks
├── src/             # Feature engineering and model logic
├── requirements.txt
└── README.md
```

---

## Data

* **League:** English Premier League
* **Granularity:** Match-level
* **Target Variable:** Match outcome (win / draw / loss)
* **Feature Types:**

  * Team performance statistics
  * Rolling form indicators
  * Venue effects (home vs. away)
  * Opponent-relative metrics

All features are constructed with **temporal leakage prevention** to ensure valid evaluation.

---

## Machine Learning Pipeline

The modeling workflow follows a structured, reproducible approach:

1. **Data Cleaning & Preprocessing**

   * Handling missing values
   * Encoding categorical variables
   * Chronological train/test splitting

2. **Feature Engineering**

   * Lagged and rolling statistics
   * Team strength and recent form metrics
   * Home-field adjustments

3. **Modeling**

   * Logistic regression baseline
   * Feature selection and regularization
   * Exploration of tree-based models

4. **Evaluation**

   * Classification metrics (accuracy, log loss)
   * Baseline comparison
   * Error analysis and diagnostics

---

## Key Insights

* Strong feature engineering outweighs model complexity
* Simple baselines provide interpretability and stability
* Improper validation leads to severe overestimation
* Sports prediction benefits from disciplined temporal modeling

---

## Why This Project Matters

This repository demonstrates:

* Applied machine learning on real-world data
* Awareness of common ML pitfalls (leakage, overfitting)
* Clean project organization and reproducibility
* Domain-specific feature engineering intuition

It mirrors how ML problems are approached in **production analytics and applied research teams**, not just academic settings.

---

## Future Improvements

* Incorporate player-level data and injury signals
* Add gradient boosting and neural network models
* Perform probability calibration and uncertainty analysis
* Deploy a live prediction or inference pipeline

---

## Author

**Toye Salami**
Industrial Engineering @ Georgia Tech
Focus: Machine Learning, Data Science, Sports Analytics
