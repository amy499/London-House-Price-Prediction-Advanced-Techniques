

# London House Price Prediction: A Stacked Ensemble Approach

   

This repository contains the code and methodology for our 2nd place solution (Public Leaderboard MAE: \~152,045) in the **London House Price Prediction** Kaggle competition. Our approach utilizes advanced feature engineering, a diverse ensemble of gradient boosting models, and a Ridge Regression meta-learner to predict 2024 property prices based on historical data (1995-2023).

## 📄 Project Overview

Predicting real estate prices in a complex market like London requires handling high-dimensional data, spatio-temporal dynamics, and significant outliers. This project addresses these challenges through a robust machine learning pipeline designed to generalize well on unseen future data.

### Key Challenges Addressed

  * **Forecasting:** Predicting 2024 prices using training data ending in 2023.
  * **Skewness:** Handling extreme luxury property outliers (prices up to £100M+).
  * **Sparsity:** Managing high cardinality in categorical features like postcodes.

## 🛠️ Methodology

Our final solution implements a **two-level stacking ensemble**:

### 1\. Data Preprocessing & Feature Engineering

We focused on "Quality over Quantity," reducing an initial set of 120+ features down to \~25 high-signal attributes.

  * **Target Transformation:** Log-transformation ($log_{10}$) of the price variable to normalize the right-skewed distribution.
  * **Imputation:** Mode/Median imputation based on test set statistics to align distributions.
  * **Geospatial Features:** Calculated Euclidean distance from Central London $(51.5074^{\circ}N, -0.1278^{\circ}W)$ and created interaction terms like `lat_x_lon`.
  * **Temporal Features:** Cyclic encoding (Sine/Cosine) for sale months to capture seasonality.
  * **Property Ratios:** Engineered density metrics such as `sqm_per_room` and `bed_bath_ratio`.

### 2\. Level-0 Models (Base Learners)

We trained **15 distinct Gradient Boosting models** using 10-Fold Stratified Cross-Validation (stratified by price deciles):

  * **9 CatBoost Regressors:** Varied by depth (7-9), learning rate, regularization, and bagging temperature.
  * **6 LightGBM Regressors:** Varied by depth (8-10), num\_leaves, and subsampling parameters.

*Note: All models used GPU acceleration for efficient training.*

### 3\. Level-1 Meta-Model (Stacking)

The predictions from the 15 base models were combined using **Ridge Regression** ($\alpha=0.01$). This method outperformed simple averaging and inverse-MAE weighting by effectively handling multicollinearity among the base predictors.

## 📊 Results

| Stacking Strategy | OOF MAE (Log Scale) |
| :--- | :--- |
| Simple Average | 0.073299 |
| Inverse MAE Weighting | 0.073299 |
| **Ridge Regression** | **0.072777** |

**Final Public Leaderboard MAE:** 152,045.07

## 💻 Installation & Usage

### Prerequisites

  * Python 3.8+
  * GPU recommended for CatBoost/LightGBM training

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/london-house-price-prediction.git
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn catboost lightgbm
    ```

### Running the Solution

1.  Ensure `train.csv` and `test.csv` are in the root directory.
2.  Run the final submission script:
    ```bash
    python submission_step10_final_push.py
    ```
3.  The script will generate `submission_step10_final_push.csv` containing the predicted prices for the test set.

## 📂 Repository Structure

  * `submission.py`: The complete end-to-end training and inference pipeline.

## 🧠 Key Insights

  * **Gradient Boosting Dominance:** Tree-based models (CatBoost/LGBM) significantly outperformed Deep Learning (FT-Transformer) and Random Forests on this tabular dataset.
  * **Feature Engineering \> Architecture:** Curating domain-specific features (geospatial/ratios) provided larger performance gains than hyperparameter tuning.
  * **Validation Strategy:** Stratified K-Fold based on price quantiles was crucial for stabilizing predictions against market volatility.



  * Prokhorenkova, L., et al. (2018). *CatBoost: Unbiased boosting with categorical features*. NeurIPS.
  * Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS.
  * Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*.
