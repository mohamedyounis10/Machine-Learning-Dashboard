import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    mean_squared_error,
)
from sklearn.cluster import KMeans
from kneed import KneeLocator
from scipy import stats

# ----------------------------------------------------
# Custom Page Styling
# ----------------------------------------------------
st.set_page_config(page_title="AI Auto ML Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2c3e50;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Header
# ----------------------------------------------------
st.markdown('<div style="text-align:center;"><h1>📊 AI Auto Machine Learning Dashboard</h1></div>', unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'>Upload data → Explore it → Train models → Visualize results</div>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# ----------------------------------------------------
# If no file
# ----------------------------------------------------
if uploaded_file is None:
    st.info("📌 Please upload a CSV file from the sidebar to start.")
else:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())
    
    target_col = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    
    # ----------------------------------------------------
    # Data Cleaning
    # ----------------------------------------------------
    st.subheader("🧼 Data Preprocessing")
    
    # Missing Values
    missing_before = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if col in numeric_cols:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    
    missing_after = df.isnull().sum().sum()
    st.success(f"Missing values handled. Before: {missing_before}, After: {missing_after}")
    
    # Remove Duplicates
    before_dup = df.shape[0]
    df.drop_duplicates(inplace=True)
    after_dup = df.shape[0]
    st.info(f"Removed {before_dup - after_dup} duplicate rows.")
    
    # Outliers
    if st.checkbox("Remove Outliers (Z-Score)"):
        before_shape = df.shape
        numeric_cols_for_outliers = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols_for_outliers) > 0:
            df = df[(np.abs(stats.zscore(df[numeric_cols_for_outliers])) < 3).all(axis=1)]
            after_shape = df.shape
            st.success(f"Outliers removed. Before: {before_shape}, After: {after_shape}")
    
    # Encoding
    st.subheader("🔤 Encoding Categorical Features")
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Encode target if categorical
    target_encoder = None
    if target_col in categorical_cols:
        target_encoder = LabelEncoder()
        df[target_col] = target_encoder.fit_transform(df[target_col])
        categorical_cols.remove(target_col)
        st.info(f"Target column '{target_col}' encoded.")
    
    for col in categorical_cols:
        if df[col].nunique() <= 10:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
            st.info(f"OneHot Encoding applied → {col}")
        else:
            df[col] = LabelEncoder().fit_transform(df[col])
            st.info(f"Label Encoding applied → {col}")
    
    # ----------------------------------------------------
    # EDA
    # ----------------------------------------------------
    st.subheader("📈 Exploratory Data Analysis")
    
    if st.checkbox("Show Correlation Heatmap"):
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        if numeric_df.shape[1] > 1:
            fig = plt.figure(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=False, cmap="viridis")
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation heatmap.")
    
    if st.checkbox("Show Histogram"):
        col = st.selectbox("Select Column", df.columns)
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)
    
    if st.checkbox("Show Boxplot"):
        numeric_cols_box = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if len(numeric_cols_box) > 0:
            col = st.selectbox("Select column for Boxplot", numeric_cols_box, key="box")
            fig = px.box(df, y=col)
            st.plotly_chart(fig)
        else:
            st.warning("No numeric columns available for boxplot.")
    
    # ----------------------------------------------------
    # Splitting Data
    # ----------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Get numeric columns after encoding
    numeric_cols_final = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Scaling
    if len(numeric_cols_final) > 0:
        scaler = StandardScaler()
        X_train[numeric_cols_final] = scaler.fit_transform(X_train[numeric_cols_final])
        X_test[numeric_cols_final] = scaler.transform(X_test[numeric_cols_final])
        st.success("Scaling applied to numeric features.")
    
    # PCA
    apply_pca = False
    if X_train.shape[1] > 10:
        pca = PCA(n_components=min(10, X_train.shape[0] - 1))
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        apply_pca = True
        st.success(f"PCA applied: Reduced to {pca.n_components_} components.")
    else:
        X_train_pca = X_train.values
        X_test_pca = X_test.values
    
    # ----------------------------------------------------
    # Analysis Type
    # ----------------------------------------------------
    analysis = st.sidebar.radio("Select Analysis Type", ["Classification", "Regression", "Unsupervised"])
    
    # ----------------------------------------------------
    # Classification
    # ----------------------------------------------------
    if analysis == "Classification":
        st.subheader("🧮 Classification Models")
        models = st.multiselect("Choose Models", ["Naive Bayes", "Decision Tree"])
        metrics = st.multiselect("Show Metrics", ["Classification Report", "Confusion Matrix"])
        
        accuracy_dict = {}
        
        for m in models:
            if m == "Naive Bayes":
                model = GaussianNB()
            else:
                model = DecisionTreeClassifier()
            
            with st.spinner(f"Training {m}..."):
                model.fit(X_train_pca, y_train)
                pred = model.predict(X_test_pca)
                accuracy_dict[m] = accuracy_score(y_test, pred)
            
            st.write(f"### 📌 {m} Results")
            st.write(f"**Accuracy:** {accuracy_dict[m]:.4f}")
            
            if "Classification Report" in metrics:
                st.text(classification_report(y_test, pred))
            
            if "Confusion Matrix" in metrics:
                cm = confusion_matrix(y_test, pred)
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               labels=dict(x="Predicted", y="Actual"))
                st.plotly_chart(fig)
        
        if accuracy_dict:
            st.subheader("📊 Accuracy Comparison")
            fig = px.bar(x=list(accuracy_dict.keys()), y=list(accuracy_dict.values()),
                        labels={"x": "Model", "y": "Accuracy"})
            st.plotly_chart(fig)
    
    # ----------------------------------------------------
    # Regression
    # ----------------------------------------------------
    if analysis == "Regression":
        st.subheader("📐 Regression Models")
        models = st.multiselect("Choose Models", ["Linear Regression", "KNN Regressor"])
        
        mse_dict = {}
        
        for m in models:
            if m == "Linear Regression":
                model = LinearRegression()
            else:
                model = KNeighborsRegressor()
            
            with st.spinner(f"Training {m}..."):
                model.fit(X_train_pca, y_train)
                pred = model.predict(X_test_pca)
                mse_dict[m] = mean_squared_error(y_test, pred)
            
            st.write(f"### 📌 {m} Results")
            st.write(f"**MSE:** {mse_dict[m]:.4f}")
            
            fig = px.scatter(x=y_test, y=pred,
                           labels={"x": "True Values", "y": "Predicted Values"},
                           title=f"{m} - Actual vs Predicted")
            fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                    y=[y_test.min(), y_test.max()],
                                    mode='lines', name='Perfect Prediction',
                                    line=dict(dash='dash', color='red')))
            st.plotly_chart(fig)
        
        if mse_dict:
            st.subheader("📊 MSE Comparison")
            fig = px.bar(x=list(mse_dict.keys()), y=list(mse_dict.values()),
                        labels={"x": "Model", "y": "MSE"})
            st.plotly_chart(fig)
    
    # ----------------------------------------------------
    # Unsupervised
    # ----------------------------------------------------
    if analysis == "Unsupervised":
        st.subheader("🔀 KMeans Clustering")
        
        # Use original data for clustering (not split)
        X_cluster = df.drop(columns=[target_col])
        
        # Scale if needed
        if len(numeric_cols_final) > 0:
            scaler_cluster = StandardScaler()
            X_cluster[numeric_cols_final] = scaler_cluster.fit_transform(X_cluster[numeric_cols_final])
        
        # Reduce dimensions if too many features
        if X_cluster.shape[1] > 2:
            pca_cluster = PCA(n_components=2)
            X_cluster_2d = pca_cluster.fit_transform(X_cluster)
        else:
            X_cluster_2d = X_cluster.values
        
        # Elbow method
        wcss = []
        K = range(2, min(11, len(X_cluster)))
        
        with st.spinner("Finding optimal number of clusters..."):
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_cluster)
                wcss.append(kmeans.inertia_)
        
        # Find elbow
        try:
            elbow = KneeLocator(list(K), wcss, curve="convex", direction="decreasing").elbow
            if elbow is None:
                elbow = 3
        except:
            elbow = 3
        
        st.write(f"### ✅ Optimal Number of Clusters: {elbow}")
        
        # Plot elbow curve
        fig = px.line(x=list(K), y=wcss, labels={"x": "Number of Clusters (k)", "y": "WCSS"},
                     title="Elbow Method")
        fig.add_vline(x=elbow, line_dash="dash", line_color="red", 
                     annotation_text=f"Optimal k={elbow}")
        st.plotly_chart(fig)
        
        # Perform clustering with optimal k
        kmeans_final = KMeans(n_clusters=elbow, random_state=42, n_init=10)
        labels = kmeans_final.fit_predict(X_cluster)
        
        # Visualize clusters
        cluster_df = pd.DataFrame(X_cluster_2d, columns=['Component 1', 'Component 2'])
        cluster_df['Cluster'] = labels.astype(str)
        
        fig = px.scatter(cluster_df, x='Component 1', y='Component 2', color='Cluster',
                        title=f"KMeans Clustering (k={elbow})")
        st.plotly_chart(fig)