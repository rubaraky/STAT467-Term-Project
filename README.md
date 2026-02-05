# Multivariate Analysis of Dry Bean Varieties based on Morphological Characteristics

![Project Status](https://img.shields.io/badge/Status-Completed-green)
![Course](https://img.shields.io/badge/Course-STAT467-blue)
![Language](https://img.shields.io/badge/Language-Python-yellow)

##  Project Overview
**Group 23** This project performs a comprehensive **Multivariate Statistical Analysis** on the Dry Bean Dataset. The primary objective is to classify and discriminate seven registered dry bean varieties based on 16 quantitative morphological features (dimensions and shape forms) using various statistical techniques.

The study aims to answer the research question: *Can we accurately distinguish bean species solely based on their numerical geometric features?*

##  Team Members
* **Ali Altuntaş**
* **Rubar Akyıldız**
* **Melih Akay**

---

##  Dataset
* **Source:** UCI Machine Learning Repository
* **Sample Size:** 13,611 observations
* **Classes:** 7 Bean Varieties (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira)
* **Features:** 16 Attributes (Area, Perimeter, Major/Minor Axis Length, Aspect Ratio, Eccentricity, Convex Area, EquivDiameter, Extent, Solidity, Roundness, Compactness, ShapeFactors 1-4)

---

## Methodology & Analysis Steps

The project follows a structured statistical workflow:

### 1. Data Cleaning & Preparation
* Handling duplicates.
* Outlier detection (Outliers were retained as they represent natural biological variation, specifically for the *Bombay* variety).
* Stratified Random Sampling for Train/Test splitting.

### 2. Exploratory Data Analysis (EDA)
* Correlation Matrices to detect multicollinearity.
* Boxplots and Histograms for distribution analysis.
* **Assumptions Check:** Multivariate Normality and Homogeneity of Covariance Matrices.

### 3. Dimensionality Reduction
* **PCA (Principal Component Analysis):** Reduced 16 variables to 3 components explaining >80% variance.
* **Factor Analysis:** Identified 3 interpretable factors:
    1.  **Size** (Area, Perimeter, Axis Lengths)
    2.  **Shape** (Compactness, Roundness)
    3.  **Solidity** (Compactness, Solidity)

### 4. Hypothesis Testing
* **MANOVA:** Confirmed that mean vectors of bean varieties are significantly different ($p < 0.05$).
* **Hotelling's $T^2$:** Used for post-hoc pairwise comparisons (e.g., Seker vs. Dermason).

### 5. Classification (Predictive Modeling)
* Comparision of **LDA** (Linear Discriminant Analysis) vs. **QDA** (Quadratic Discriminant Analysis).
* **Selected Model:** **QDA** was chosen due to heterogeneity of covariance matrices.

### 6. Clustering (Unsupervised Learning)
* **K-Means Clustering:** Applied with $k=7$ to validate biological labels against geometric natural grouping.

### 7. Canonical Correlation Analysis (CCA)
* Analyzed the relationship between the **Size** variable set and the **Shape** variable set.

---

##  Key Results

| Metric | Result |
| :--- | :--- |
| **QDA Accuracy** | **90.5%** |
| **Dominant Clusters** | 3 Major Physical Groups (despite 7 labels) |
| **Canonical Correlation** | **0.99** (Near-perfect link between Size & Shape) |

###  Critical Insights
1.  **Classification:** The model achieved high accuracy. The *Bombay* variety was classified with 100% success. The main confusion occurred between *Sira* and *Dermason* due to their morphological similarity.
2.  **Clustering:** Although there are 7 biological species, K-Means revealed that they morphologically collapse into **3 main groups** (Small, Medium, Large).
3.  **Variable Redundancy:** CCA showed that Size and Shape are strictly dependent; knowing the dimensions allows for a near-perfect prediction of the shape.

---

##  Technologies & Libraries
* **Python** (3.x)
* **Pandas & NumPy:** Data manipulation.
* **Matplotlib & Seaborn:** Visualization.
* **Scikit-learn:** PCA, Clustering, Classification metrics.
* **SciPy / Statsmodels:** Statistical tests (MANOVA, Canonical Correlation).

##  How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/dry-bean-multivariate-analysis.git](https://github.com/yourusername/dry-bean-multivariate-analysis.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn scipy
    ```
3.  Run the analysis script (or Jupyter Notebook):
    ```bash
    jupyter notebook Project_Notebook.ipynb
    ```

## References
* *Koklu, M. and Ozkan, I.A., (2020), "Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques." Computers and Electronics in Agriculture, 174, 105507.*
* UCI Machine Learning Repository: Dry Bean Dataset.

---
*Created for ODTÜ (METU) STAT 467 - Fall 2026*
