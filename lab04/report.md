# Lab 04 Report

This report documents the work completed in Lab 04.

## Summary
1. Dataset and Problem Definition: Fetched the UCI Adult dataset using fetch_openml, outlining a half-page problem definition focused on 14 input features, predicting whether income exceeds 50K, tracking F1 score and accuracy, and maintaining a strict no-leakage reproducible pipeline.
2. Data Ingestion Step: Ingested the raw dataset without modifications, logging an initial row count of 48,842 and 15 columns (including target).
3. Validation Step: Defined and executed explicit schema mappings, range boundaries (age 17–90, hours 1–99, etc.), missing-value policy handling '?' markers explicitly without dropping rows, and reporting violations.
4. Cleaning and Transformation: Implemented a leakage-free preprocessing pipeline by applying a stratified 70/30 train/test split prior to cleaning, fitting imputers (mode for categorical, median for numeric) and outlier clipping (1st/99th percentiles) on the training set only, dropping exact duplicates from training, and serializing the transformer via joblib.dump.
5. Feature Engineering: Engineered two derived features (capital_net and hours_per_age), evaluating and recording Logistic Regression baseline performance metrics with and without them.
6. Label-Quality Check: Performed a duplicate-conflict audit across sampled rows to identify conflicting income labels for identical feature vectors, and manually validated records against impossible value constraints.
7. Artifact Generation: Calculated the SHA256 cryptographic hash of the processed dataset, stored configuration parameters in a YAML file, generated a comprehensive data_card.md, and ensured end-to-end single-command reproducibility.

## Notes
Pipeline Integrity: Enforcing strict separation between the training and test sets during imputation, outlier bounds calculation, and duplicate dropping prevents data leakage and ensures that generalization metrics remain unbiased.
Handling Missing Values: Rather than silently discarding rows containing unknown markers (?), treating them as categorical tokens or handling them through explicit missing indicators preserves valuable dataset volume while satisfying robustness constraints.
Feature Impact: Derived features such as net capital and workload intensity relative to age can provide linear models with useful nonlinear combinations, though regularized models like Logistic Regression require proper scaling to prevent magnitude dominance.
