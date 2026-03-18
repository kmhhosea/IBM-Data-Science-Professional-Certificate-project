# SpaceX Falcon 9 First Stage Landing Prediction

**Author:** Anik Tahabilder
**Department:** Computer Science, Wayne State University
**Course:** IBM Data Science Professional Certificate - Applied Data Science Capstone

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Problem](#business-problem)
3. [Dataset Overview](#dataset-overview)
4. [Methodology](#methodology)
5. [Data Collection](#data-collection)
6. [Exploratory Data Analysis](#exploratory-data-analysis)
7. [Interactive Visual Analytics](#interactive-visual-analytics)
8. [Machine Learning Prediction](#machine-learning-prediction)
9. [Model Comparison and Selection](#model-comparison-and-selection)
10. [Key Findings](#key-findings)
11. [Conclusion](#conclusion)
12. [How to Run This Project](#how-to-run-this-project)
13. [Project Structure](#project-structure)

---

## Executive Summary

This project predicts whether SpaceX's Falcon 9 first stage rocket will successfully land after launch. SpaceX advertises Falcon 9 rocket launches at $62 million, significantly cheaper than competitors charging $165+ million per launch. This cost advantage stems from SpaceX's ability to reuse the first stage booster. By predicting landing success, we can estimate launch costs and provide valuable insights for companies bidding against SpaceX for rocket launch contracts.

**Key Result:** The **Decision Tree Classifier** achieved the best cross-validation accuracy of **87.5%** with test accuracy reaching **88.89-94.44%**, making it the recommended model for predicting Falcon 9 first stage landing outcomes.

---

## Business Problem

### Problem Statement
SpaceX's revolutionary approach to rocket reusability has disrupted the space launch industry. If we can accurately predict whether the Falcon 9 first stage will land successfully, we can:

1. **Estimate launch costs** - A successful landing means the booster can be reused, reducing costs
2. **Support competitive bidding** - Alternative companies can make informed bids against SpaceX
3. **Understand success factors** - Identify which variables most influence landing success

### Why This Matters
- **Cost Difference:** ~$103 million savings per launch when first stage is recovered
- **Industry Impact:** Enables competitors to better understand SpaceX's cost structure
- **Data-Driven Decisions:** Provides quantifiable predictions for launch cost estimation

---

## Dataset Overview

### Data Sources
1. **SpaceX REST API** - Primary source for launch data
2. **Wikipedia Web Scraping** - Supplementary launch records

### Dataset Characteristics

| Attribute | Description |
|-----------|-------------|
| **Records** | 90 Falcon 9 launches (filtered from 94 total) |
| **Date Range** | June 2010 - November 2020 |
| **Target Variable** | `Class` (1 = Successful Landing, 0 = Failed Landing) |
| **Features** | 83 features after one-hot encoding |

### Key Features

| Feature | Type | Description |
|---------|------|-------------|
| `FlightNumber` | Numeric | Sequential launch number |
| `PayloadMass` | Numeric | Mass of payload in kg (Mean: ~6,104 kg) |
| `Orbit` | Categorical | Target orbit (LEO, GTO, ISS, PO, etc.) |
| `LaunchSite` | Categorical | Launch location (CCAFS SLC 40, KSC LC 39A, VAFB SLC 4E) |
| `Flights` | Numeric | Number of flights for the booster |
| `GridFins` | Boolean | Whether grid fins were used |
| `Reused` | Boolean | Whether the booster was previously used |
| `Legs` | Boolean | Whether landing legs were deployed |
| `Block` | Numeric | Booster block version (1-5) |
| `ReusedCount` | Numeric | Number of times booster was reused |

### Data Preprocessing
- **Missing Values:** `PayloadMass` nulls (5 records) imputed with mean value
- **Encoding:** One-hot encoding applied to categorical variables (Orbit, LaunchSite, LandingPad, Serial)
- **Standardization:** StandardScaler applied for ML model training
- **Train/Test Split:** 80/20 split with `random_state=2`

---

## Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

```
1. Business Understanding    -> Define prediction goal and success criteria
2. Data Understanding        -> Collect and explore SpaceX launch data
3. Data Preparation          -> Clean, transform, and engineer features
4. Modeling                  -> Train and tune multiple ML algorithms
5. Evaluation                -> Compare models using cross-validation
6. Deployment                -> Document findings and recommendations
```

### Tools and Technologies

| Category | Tools |
|----------|-------|
| **Programming** | Python 3.x |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Folium |
| **Machine Learning** | Scikit-learn |
| **Database** | SQLite, SQL |
| **Web Scraping** | BeautifulSoup, Requests |

---

## Data Collection

### 1. SpaceX API Data Collection
**Notebook:** `jupyter-labs-spacex-data-collection-api.ipynb`

```python
# API Endpoint
spacex_url = "https://api.spacexdata.com/v4/launches/past"

# Data extraction workflow:
# 1. GET request to SpaceX API
# 2. Parse JSON response with pd.json_normalize()
# 3. Extract booster version, launch site, payload, and core data
# 4. Filter for Falcon 9 launches only
# 5. Handle missing values
```

**Key API Endpoints Used:**
- `/v4/launches/past` - Historical launch data
- `/v4/rockets/{id}` - Booster version details
- `/v4/launchpads/{id}` - Launch site coordinates
- `/v4/payloads/{id}` - Payload mass and orbit
- `/v4/cores/{id}` - Core reuse information

### 2. Web Scraping Wikipedia
**Notebook:** `jupyter-labs-webscraping.ipynb`

```python
# Target URL (snapshot from June 9, 2021)
static_url = "https://en.wikipedia.org/w/index.php?title=List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922"

# Extraction process:
# 1. HTTP GET request with custom headers
# 2. Parse HTML with BeautifulSoup
# 3. Extract table data from <th> and <td> elements
# 4. Clean and normalize data
```

**Scraped Data:** 121 launch records with flight details, outcomes, and booster landing status.

---

## Exploratory Data Analysis

### SQL Analysis
**Notebook:** `jupyter-labs-eda-sql-coursera_sqllite.ipynb`

Key SQL queries and findings:

```sql
-- Unique launch sites
SELECT DISTINCT Launch_Site FROM SPACEXTABLE;
-- Results: CCAFS LC-40, VAFB SLC-4E, KSC LC-39A, CCAFS SLC-40

-- Total payload mass for NASA (CRS) missions
SELECT SUM(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Customer = 'NASA (CRS)';
-- Result: 45,596 kg

-- Average payload for F9 v1.1
SELECT AVG(PAYLOAD_MASS__KG_) FROM SPACEXTABLE WHERE Booster_Version = 'F9 v1.1';
-- Result: 2,928.4 kg

-- First successful ground pad landing
SELECT MIN(Date) FROM SPACEXTABLE WHERE Landing_Outcome = 'Success (ground pad)';
-- Result: 2015-12-22
```

### Visualization Analysis
**Notebook:** `jupyter-labs-eda-dataviz-v2.ipynb`

#### Key Visualizations Created:
1. **Flight Number vs. Launch Site** - Launch frequency patterns
2. **Payload Mass vs. Launch Site** - Payload capacity by location
3. **Success Rate by Orbit Type** - Orbit-specific success rates
4. **Yearly Success Trend** - Improvement over time

#### Key Insights:
- **Success rate increased** significantly after 2015
- **KSC LC-39A** has the highest success rate among launch sites
- **LEO and ISS orbits** show strong correlation between flight number and success
- **Heavy payloads (>10,000 kg)** are not launched from VAFB-SLC

---

## Interactive Visual Analytics

### Folium Maps Analysis
**Notebook:** `lab-jupyter-launch-site-location-v2.ipynb`

Interactive maps were created to analyze:

1. **Launch Site Locations**
   - All sites are in proximity to the equator (favorable for orbital mechanics)
   - All sites are close to coastlines (safety for failed launches)

2. **Success/Failure Markers**
   - Color-coded markers (green=success, red=failure)
   - Cluster visualization for overlapping launches

3. **Proximity Analysis**
   - Distance to coastline
   - Distance to railways, highways, and cities
   - Launch sites maintain safe distances from populated areas

**Launch Sites Analyzed:**

| Site | Location | Coordinates |
|------|----------|-------------|
| CCAFS LC-40 | Cape Canaveral, FL | 28.56°N, 80.58°W |
| CCAFS SLC-40 | Cape Canaveral, FL | 28.56°N, 80.58°W |
| KSC LC-39A | Kennedy Space Center, FL | 28.61°N, 80.60°W |
| VAFB SLC-4E | Vandenberg AFB, CA | 34.63°N, 120.61°W |

---

## Machine Learning Prediction

**Notebook:** `SpaceX-Machine-Learning-Prediction-Part-5-v1.ipynb`

### Data Preparation

```python
# Feature matrix: 83 features after one-hot encoding
X = pd.read_csv('dataset_part_3.csv')

# Target variable
Y = data['Class'].to_numpy()

# Standardization
transform = preprocessing.StandardScaler()
X = transform.fit_transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# Training samples: 72, Test samples: 18
```

### Models Evaluated

Four classification algorithms were evaluated using **GridSearchCV** with **10-fold cross-validation**:

#### 1. Logistic Regression

```python
parameters = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}
```

**Best Parameters:** `C=0.01, penalty='l2', solver='lbfgs'`
**Cross-Validation Accuracy:** 84.64%
**Test Accuracy:** 83.33%

#### 2. Support Vector Machine (SVM)

```python
parameters = {
    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    'C': np.logspace(-3, 3, 5),
    'gamma': np.logspace(-3, 3, 5)
}
```

**Best Parameters:** `C=1.0, gamma=0.0316, kernel='sigmoid'`
**Cross-Validation Accuracy:** 84.82%
**Test Accuracy:** 83.33%

#### 3. Decision Tree Classifier

```python
parameters = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 6, 8, 10, 12, 14, 16, 18],
    'max_features': ['sqrt', 'log2', None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10]
}
```

**Best Parameters:**
- `criterion='entropy'`
- `max_depth=6`
- `max_features='log2'`
- `min_samples_leaf=1`
- `min_samples_split=5`
- `splitter='best'`

**Cross-Validation Accuracy:** 87.5%
**Test Accuracy:** 88.89-94.44%

#### 4. K-Nearest Neighbors (KNN)

```python
parameters = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}
```

**Best Parameters:** `algorithm='auto', n_neighbors=6, p=1`
**Cross-Validation Accuracy:** 84.46%
**Test Accuracy:** 94.44%

---

## Model Comparison and Selection

### Performance Summary

| Model | CV Accuracy | Test Accuracy | Precision | Recall | F1-Score |
|-------|-------------|---------------|-----------|--------|----------|
| **Decision Tree** | **87.50%** | **88.89%** | 1.000 | 0.857 | 0.923 |
| SVM | 84.82% | 83.33% | 0.929 | 0.929 | 0.929 |
| Logistic Regression | 84.64% | 83.33% | 0.933 | 1.000 | 0.966 |
| KNN | 84.46% | 94.44% | 0.933 | 1.000 | 0.966 |

### Why Decision Tree Was Selected

The **Decision Tree Classifier** was selected as the best model for the following reasons:

#### 1. Highest Cross-Validation Accuracy (87.5%)
- Most reliable indicator of generalization performance
- Less susceptible to overfitting on small test sets

#### 2. Interpretability
- Decision rules can be visualized and understood
- Important for stakeholders who need to explain predictions
- Clear feature importance rankings

#### 3. No Feature Scaling Required
- Works with both numerical and categorical features natively
- Less preprocessing complexity

#### 4. Handles Non-Linear Relationships
- Captures complex interactions between features
- Orbit type, payload mass, and booster version interact non-linearly

### Why Not Other Models?

| Model | Reason Not Selected |
|-------|---------------------|
| **Logistic Regression** | Lower CV accuracy (84.64%); assumes linear relationships which may not hold for rocket landing physics |
| **SVM** | Similar CV accuracy but less interpretable; sigmoid kernel may not capture all patterns |
| **KNN** | High test accuracy but lower CV accuracy (84.46%); computationally expensive at prediction time; sensitive to feature scaling |

### Confusion Matrix Analysis (Decision Tree)

```
                 Predicted
              |  No Land  |  Landed  |
Actual -------|-----------|----------|
  No Land     |     3     |    0     |  (True Negatives, False Positives)
  Landed      |     2     |   13     |  (False Negatives, True Positives)
```

- **Strength:** Zero false positives (never incorrectly predicts success)
- **Area for Improvement:** Some false negatives (missed successful landings)

---

## Key Findings

### Technical Findings

1. **Flight Experience Matters**
   - Success rate increases with flight number
   - Later launches (post-2015) show significantly higher success rates

2. **Launch Site Influence**
   - KSC LC-39A has the highest success rate
   - All sites are strategically located near coastlines

3. **Payload Mass Impact**
   - Moderate payloads (2,000-6,000 kg) show good success rates
   - Very heavy payloads have mixed results depending on orbit

4. **Booster Reuse Success**
   - Reused boosters with grid fins and legs show high success rates
   - Block 5 boosters demonstrate the best performance

### Business Implications

1. **Cost Estimation:** Successful landing prediction enables accurate cost modeling
2. **Risk Assessment:** Identify high-risk launch configurations
3. **Competitive Intelligence:** Understand SpaceX's operational patterns

---

## Conclusion

This capstone project successfully demonstrates the application of data science methodologies to a real-world aerospace problem. By collecting data from multiple sources, performing thorough exploratory analysis, and training multiple machine learning models, we achieved:

- **Prediction Accuracy:** 87.5% cross-validation accuracy with Decision Tree
- **Actionable Insights:** Identified key factors influencing landing success
- **Business Value:** Enabled launch cost estimation for competitive bidding

The Decision Tree model provides the best balance of accuracy and interpretability for predicting Falcon 9 first stage landing success.

---

## How to Run This Project

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn folium beautifulsoup4 requests sqlalchemy
```

### Execution Order

Run the notebooks in the following sequence:

1. **Data Collection**
   ```
   jupyter-labs-spacex-data-collection-api.ipynb
   jupyter-labs-webscraping.ipynb
   ```

2. **Exploratory Data Analysis**
   ```
   jupyter-labs-eda-sql-coursera_sqllite.ipynb
   jupyter-labs-eda-dataviz-v2.ipynb
   ```

3. **Interactive Analytics**
   ```
   lab-jupyter-launch-site-location-v2.ipynb
   ```

4. **Machine Learning**
   ```
   SpaceX-Machine-Learning-Prediction-Part-5-v1.ipynb
   ```

---

## Project Structure

```
IBM-Data-Science-Certificate/
|
|-- jupyter-labs-spacex-data-collection-api.ipynb  # API data collection
|-- jupyter-labs-webscraping.ipynb                  # Web scraping from Wikipedia
|-- jupyter-labs-eda-sql-coursera_sqllite.ipynb    # SQL-based analysis
|-- jupyter-labs-eda-dataviz-v2.ipynb              # EDA with visualizations
|-- lab-jupyter-launch-site-location-v2.ipynb      # Folium interactive maps
|-- SpaceX-Machine-Learning-Prediction-Part-5-v1.ipynb  # ML models
|
|-- dataset/                                        # Data files directory
|-- figures/                                        # Generated visualizations
|-- README.md                                       # This file
```

---

## Acknowledgments

- **IBM Skills Network** - Course materials and lab infrastructure
- **SpaceX** - Public API providing launch data
- **Wikipedia** - Historical launch records

---

## Contact

**Anik Tahabilder**
Department of Computer Science
Wayne State University

---

*This project was completed as part of the IBM Data Science Professional Certificate program.*
