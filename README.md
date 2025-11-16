# 📊 AI Auto Machine Learning Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**An intelligent web-based dashboard for automated machine learning workflows**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Screenshots](#-screenshots) • [Demo](#-demo)

</div>

---

## 📸 Screenshots

<div align="center">
<img width="1919" height="1022" alt="Screenshot 2025-11-16 163052" src="https://github.com/user-attachments/assets/8742a49d-cd36-48eb-bf84-6194571f4dd7" />

</div>

---

## 🎥 Demo Video

<!-- Add your demo video here -->
<div align="center">

[![Demo Video](https://img.shields.io/badge/▶️-Watch%20Demo-red?style=for-the-badge)](https://your-video-link-here.com)

*Replace the link above with your actual demo video URL (YouTube, Vimeo, etc.)*

</div>

---

## 📖 Overview

**AI Auto Machine Learning Dashboard** is a comprehensive web application built with Streamlit that automates the entire machine learning pipeline. From data preprocessing to model training and evaluation, this dashboard provides an intuitive interface for both beginners and experienced data scientists.

### ✨ Key Highlights

- 🚀 **Fully Automated Pipeline**: Upload data and get results in minutes
- 🎯 **Multiple ML Tasks**: Classification, Regression, and Clustering
- 📊 **Interactive Visualizations**: Beautiful charts and graphs using Plotly
- 🧹 **Smart Preprocessing**: Automatic handling of missing values, encoding, and scaling
- 📈 **Exploratory Data Analysis**: Built-in EDA tools for data insights

---

## 🎯 Features

### 🔧 Data Preprocessing
- ✅ **Missing Values Handling**: Automatic imputation for numeric and categorical features
- ✅ **Duplicate Removal**: Clean your dataset with one click
- ✅ **Outlier Detection**: Z-score based outlier removal
- ✅ **Feature Encoding**: Automatic Label Encoding and One-Hot Encoding
- ✅ **Feature Scaling**: StandardScaler for numeric features
- ✅ **Dimensionality Reduction**: Automatic PCA when features > 10

### 📊 Exploratory Data Analysis
- 📈 Correlation Heatmap
- 📊 Histograms
- 📦 Boxplots
- 🔍 Dataset Statistics

### 🤖 Machine Learning Models

#### Classification
- 🎯 **Naive Bayes**: Gaussian Naive Bayes classifier
- 🌳 **Decision Tree**: Decision Tree classifier
- 📊 **Metrics**: Accuracy, Classification Report, Confusion Matrix

#### Regression
- 📐 **Linear Regression**: Standard linear regression
- 🔢 **KNN Regressor**: K-Nearest Neighbors regression
- 📊 **Metrics**: MSE, Actual vs Predicted visualizations

#### Unsupervised Learning
- 🔀 **KMeans Clustering**: Automatic optimal cluster detection using Elbow Method
- 📊 **Visualizations**: 2D cluster visualization with PCA

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/ai-auto-ml-dashboard.git
cd ai-auto-ml-dashboard
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

If `requirements.txt` doesn't exist, install manually:
```bash
pip install streamlit pandas numpy plotly seaborn matplotlib scikit-learn kneed scipy
```

### Step 3: Run the Application
```bash
streamlit run app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

---

## 💻 Usage

### Quick Start Guide

1. **Upload Your Dataset**
   - Click on "Upload CSV File" in the sidebar
   - Select your CSV file

2. **Select Target Column**
   - Choose the column you want to predict/analyze

3. **Data Preprocessing** (Automatic)
   - The dashboard automatically handles missing values and duplicates
   - Optionally remove outliers using the checkbox

4. **Choose Analysis Type**
   - **Classification**: For categorical target variables
   - **Regression**: For continuous target variables
   - **Unsupervised**: For clustering without a target

5. **Select Models**
   - Choose one or more models to train
   - View results and visualizations

6. **Explore Results**
   - Check accuracy/MSE metrics
   - View confusion matrices
   - Analyze visualizations

---

## 📁 Project Structure

```
ai-auto-ml-dashboard/
│
├── app.py                          # Main Streamlit application
├── notebook.ipynb                  # Jupyter notebook for development
├── README.md                       # This file
│
├── Datasets/                       # Sample datasets
│   ├── Breast_cancer_data.csv
│   ├── heart.csv
│   ├── Mall_Customers.csv
│   ├── StudentPerformanceFactors.csv
│   └── Links.txt
│
└── Project Diagram.png            # Project architecture diagram
```

---

## 📊 Sample Datasets

The project includes sample datasets for testing:

1. **Student Performance Factors** (Regression)
   - Predict student performance based on various factors

2. **Heart Disease Dataset** (Classification)
   - Classify heart disease presence

3. **Breast Cancer Dataset** (Classification)
   - Predict breast cancer diagnosis

4. **Mall Customers** (Unsupervised)
   - Customer segmentation using clustering

Dataset links are available in `Datasets/Links.txt`

---

## 🛠️ Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **Seaborn & Matplotlib**: Statistical visualizations
- **Kneed**: Elbow point detection for clustering

---

## 📝 Features in Detail

### Automatic Preprocessing Pipeline
- Detects and handles missing values intelligently
- Removes duplicate rows
- Encodes categorical variables automatically
- Scales numeric features for better model performance
- Applies PCA for high-dimensional data

### Model Training & Evaluation
- Train multiple models simultaneously
- Compare model performance
- Visualize predictions vs actual values
- Generate detailed classification reports

### Interactive Visualizations
- Real-time charts and graphs
- Interactive Plotly visualizations
- Export-ready figures

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Mohamed Younis**
**Yousef Helmy**
**Ahmed Abd ElSalam**

---

## 🙏 Acknowledgments

- ITI (Information Technology Institute) for project guidance
- Streamlit team for the amazing framework
- Scikit-learn community for comprehensive ML tools

---

<div align="center">

**Made with ❤️ using Streamlit**

⭐ Star this repo if you find it helpful!

</div>

