# Predicting-House-Pricing-in-California-Using-Machine-Learning-Models
Overview
This project builds an end-to-end machine learning pipeline to predict house prices in California using demographic, geographic, and housing-related features from the 1990 California Census.The goal is to deliver a data-driven pricing decision support model that captures complex, non-linear patterns in the housing market.

Objective
- Predict median_house_value with reliable accuracy
- Identify key drivers of house prices
- Compare multiple regression models and select the best-performing approach
-Translate model performance into business impact


Dataset

Source: 1990 California Census
Observation unit: California census district
Target: median_house_value
Key features:
- Location: longitude, latitude, ocean_proximity
- Socioeconomic: median_income, population, households
- Housing: total_rooms, housing_median_age
- Engineered features: rooms_per_household, population_per_household, log-transformed variables


Key EDA Insights

- House prices are right-skewed with capped high values
- Ocean proximity and median income are the strongest predictors
- Coastal areas show significantly higher prices
- Feature–target relationships are non-linear, justifying advanced ML models


Modeling Approach

- Models evaluated: Linear Regression, KNN, Decision Tree, Random Forest, XGBoost
- Best model: XGBoost Regressor (Hyperparameter Tuned)
- Evaluation metrics: MAE, RMSE, MAPE
- Final Test Performance:
MAE: ~29,700
RMSE: ~45,700
MAPE: ~16%


Model Interpretation
Feature importance shows:
- ocean_proximity as the most influential feature
- median_income as the strongest numerical driver
  
Residual analysis indicates:
- Stable, unbiased predictions
- Higher uncertainty at very high price ranges

  
Business Impact

- Pricing error reduced from an estimated 20–25% (manual pricing) to ~16%
- Potential reduction of $10,000–20,000 price deviation per house
- Enables:
More consistent pricing decisions
Scalable, data-driven house valuation
Actionable insights into location and income effects


Recommendations

- Use the model as a decision support tool for pricing
- Apply A/B testing against manual pricing strategies
- Perform error profiling on extreme over- and underestimations
- Enhance performance with additional features (e.g. property size, accessibility)
- Retrain regularly with more recent data
- Extend the model to related use cases such as price trend prediction

  
Tech Stack
Python · pandas · numpy · seaborn · scikit-learn · XGBoost
