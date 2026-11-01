# NYC Taxi Trip Duration - Final Report

## Problem
Predict taxi trip duration in seconds.

## Dataset
Training: 1,458,644 observations
Test: 625,134 observations

## Evaluation
Root Mean Squared Logarithmic Error (RMSLE)

## Model Results

| Model | RMSLE |
|---|---:|
| HistGradientBoosting + FE | 0.411019 |
| HistGradientBoosting | 0.415086 |
| Random Forest | 0.421797 |
| Baseline | 0.812876 |

## Findings
The improved HistGradientBoosting model obtained
the lowest validation RMSLE.

Geographic and time-based features were used.
Both features and model hyperparameters changed.

## Conclusion
The final model was selected based on validation RMSLE.
The official Kaggle leaderboard score was not measured.

## Source
https://www.kaggle.com/competitions/nyc-taxi-trip-duration
