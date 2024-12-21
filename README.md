# Paid Online Advertisement CTR Prediction

## Introduction
This project analyzes the dynamics of online advertising and develops predictive models to estimate the click-through rate (CTR) of advertisements, aiding in better ad targeting and revenue generation.

## Problem Statement
Understanding and predicting CTR is essential for advertisers to optimize campaign performance, allocate budgets effectively, and maximize ad revenue.

## Methods Used
- **Data Preparation**: Shrinkage of dataset (42 million rows reduced to 1%), handling missing values, and standardizing numerical attributes.
- **Modeling**: Developed and evaluated Logistic Regression, CART, Random Forest, and XGBoost models.
- **Evaluation**: Used ROC-AUC as the primary metric, with XGBoost achieving the best performance (ROC-AUC: 0.6804).

## Results
The XGBoost model outperformed others, resulting in a 36.03% revenue increase compared to baseline predictions, emphasizing its effectiveness in real-world applications.

## Future Directions
Planned improvements include using the complete dataset, leveraging domain knowledge with raw data access, and exploring advanced models like neural networks for enhanced prediction accuracy.
