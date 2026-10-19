# House Price Prediction — King County, USA

**Course:** CISC7204 (Data Analytics) · **Student:** mc644073 Yang Yiming

A data science project that analyses and predicts residential house prices in King County (Seattle area), USA, using a public dataset of homes sold between May 2014 and May 2015.

## Problem / Research Question

Working as a data analyst for a Real Estate Investment Trust that wants to enter the residential real-estate market, the goal is:

> **Which house features best explain and predict the market price of a house, and how accurately can price be predicted from those features alone?**

The analysis covers the full pipeline — importing and wrangling the data, exploratory data analysis (EDA), feature correlation, and model development (multiple linear regression, polynomial regression, and ridge regression).

## Dataset Source and Usage Permissions

- **Source:** [King County House Sales dataset on Kaggle](https://www.kaggle.com/harlfoxem/housesalesprediction) (`kc_house_data`), slightly modified for the IBM/Coursera *Data Analysis with Python* course.
- **Loaded from:** `https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv`
- **Licence/permissions:** the dataset is publicly released for educational use (CC0 / public-domain-style release on Kaggle). It contains **no personal data** — `id` is a house identifier, not a person identifier — and the identifier columns (`id`, `Unnamed: 0`) are dropped before analysis. No passwords, API keys, or restricted data are included in this repository.

The dataset contains **21,613 rows and 20 features**, including `price`, `sqft_living`, `grade`, `sqft_above`, `bedrooms`, `bathrooms`, `waterfront`, `view`, `lat`, `long`, and `zipcode`.

## Methods and Tools Used

| Step | Details |
|---|---|
| Data wrangling | Dropped identifier columns; imputed 13 missing `bedrooms` and 10 missing `bathrooms` values with the column mean |
| EDA | Boxplots, regression plots, value counts, Pearson correlation with `price` |
| Model 1 | Simple linear regression (`sqft_living` → `price`) |
| Model 2 | Multiple linear regression on 11 selected features |
| Model 3 | Polynomial regression pipeline (`StandardScaler` + `PolynomialFeatures` + `LinearRegression`) |
| Model 4 | Ridge regression (α = 0.1) on an 85/15 train/test split |
| Model 5 | Ridge regression on degree-2 polynomial features |

**Tools:** Python 3 · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter Notebook

## Key Findings

1. **Living-space size is the dominant price driver.** `sqft_living` has the strongest correlation with price (r ≈ **0.70**), followed by overall `grade` (r ≈ 0.67) and `sqft_above` (r ≈ 0.61). A simple linear model on `sqft_living` alone reaches R² ≈ **0.49**.
2. **Waterfront location commands a large premium.** The median price of waterfront houses is roughly 2–3× that of non-waterfront houses (see `visualizations/price_by_waterfront_boxplot.png`).
3. **Model performance scales with complexity**, then saturates:

   | Model | R² |
   |---|---|
   | Simple linear (`sqft_living`) | 0.493 |
   | Multiple linear (11 features) | 0.658 |
   | Polynomial pipeline (degree 2, all 11 features) | 0.751 |
   | Ridge on train/test split (test set) | 0.648 |
   | Ridge + degree-2 polynomial features (test set) | **0.700** |

4. **Location has weak linear signal.** `long` alone explains almost nothing (R² ≈ 0.0005); latitude matters more (r ≈ 0.31) because it proxies for distance to Seattle's urban core.

**One limitation:** the models are strictly *linear/polynomial in raw tabular features* — they cannot capture neighbourhood effects captured by `zipcode`, interactions with market timing (all sales fall in one year), or non-linear structure that tree-based models (random forest, gradient boosting) would exploit. A natural next step is to compare against gradient-boosted trees and to add location encoding.

## Visualizations

| File | What it shows |
|---|---|
| `visualizations/price_by_waterfront_boxplot.png` | Price distribution by waterfront status |
| `visualizations/price_vs_sqft_above_regplot.png` | Price vs. square footage above ground, with regression line |
| `visualizations/correlation_heatmap.png` | Correlation heatmap of key features vs. price |
| `visualizations/predicted_vs_actual_ridge_poly.png` | Predicted vs. actual prices for the best ridge-polynomial model (test set) |

## Instructions for Running the Analysis

1. **Clone this repository**
   ```bash
   git clone https://github.com/YangYi-ming/data-science-project.git
   cd data-science-project
   ```
2. **Create a virtual environment and install dependencies** (Python 3.9+ recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Run the notebook.** The dataset is downloaded automatically from the public URL in the first cells, so no manual download is needed.
   ```bash
   jupyter notebook "CISC7204 Assgn01 Module 06 - Final Project mc644073 Yang Yiming.ipynb"
   ```
   Run all cells top-to-bottom (`Kernel → Restart & Run All`). Expected runtime is under 2 minutes on a typical laptop.

4. **Reproduce the extra figures** (heat map and predicted-vs-actual plot): they are generated with the same steps shown in the notebook (correlation matrix; ridge on `PolynomialFeatures(degree=2)` with `train_test_split(test_size=0.15, random_state=1)`).

> **Note:** with pandas ≥ 2.0, use `df['bedrooms'] = df['bedrooms'].replace(np.nan, df['bedrooms'].mean())` style assignments for the imputation steps (in-place `.replace()` on a column view no longer propagates in newer pandas versions). The analysis values reported above were obtained with the replacement-style imputation.
