# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Title and description
st.title("Car Price Analysis")
st.markdown("""
This application analyzes various features that impact car prices using data visualizations, descriptive statistics, and correlation analysis.
""")

# Sidebar for user input
st.sidebar.header("Dataset")
st.sidebar.write("This application uses the cleaned car dataset.")

# Load Data
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/Thanmai7717/Exam-1/refs/heads/main/Exam1_clean_df.csv'
    return pd.read_csv(url)

df = load_data()

# Display the data
st.header("Dataset Overview")
if st.checkbox("Show dataset"):
    st.write(df.head())

st.write("Data types of the columns:")
st.write(df.dtypes)

# Data Analysis Section
st.header("Data Analysis")

# Question 1: Data type of 'peak-rpm'
st.subheader("Question 1: What is the data type of the column 'peak-rpm'?")
st.write(f"The data type of 'peak-rpm' is {df['peak-rpm'].dtype}.")

# Correlation Analysis
st.subheader("Correlation Analysis")
columns_to_correlate = ['bore', 'stroke', 'compression-ratio', 'horsepower']
if st.checkbox("Show correlation matrix for specific columns"):
    st.write(df[columns_to_correlate].corr())

# Visualizations
st.header("Visualizations")

# Positive Linear Relationship
st.subheader("Positive Linear Relationship: Engine Size vs Price")
fig1, ax1 = plt.subplots()
sns.regplot(x="engine-size", y="price", data=df, ax=ax1)
st.pyplot(fig1)

# Negative Linear Relationship
st.subheader("Negative Linear Relationship: Highway-mpg vs Price")
fig2, ax2 = plt.subplots()
sns.regplot(x="highway-mpg", y="price", data=df, ax=ax2)
st.pyplot(fig2)

# Weak Linear Relationship
st.subheader("Weak Linear Relationship: Peak-rpm vs Price")
fig3, ax3 = plt.subplots()
sns.regplot(x="peak-rpm", y="price", data=df, ax=ax3)
st.pyplot(fig3)

# Boxplot for categorical variables
st.subheader("Categorical Variable Analysis")
categorical_col = st.selectbox("Select a categorical column to analyze", ['body-style', 'engine-location', 'drive-wheels'])
fig4, ax4 = plt.subplots()
sns.boxplot(x=categorical_col, y="price", data=df, ax=ax4)
st.pyplot(fig4)

# Grouping and Pivot Table
st.header("Grouping and Pivot Table")
grouping_col = st.selectbox("Select a column to group by", ['drive-wheels', 'body-style'])
grouped_df = df.groupby(grouping_col)['price'].mean().reset_index()
st.write(grouped_df)

# Heatmap for grouped data
if st.checkbox("Show heatmap for Drive-Wheels and Body-Style vs Price"):
    df_grouped = df[['drive-wheels', 'body-style', 'price']].groupby(['drive-wheels', 'body-style'], as_index=False).mean()
    pivot_table = df_grouped.pivot(index='drive-wheels', columns='body-style', values='price').fillna(0)
    fig5, ax5 = plt.subplots()
    sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="RdBu", ax=ax5)
    st.pyplot(fig5)

# Correlation and Causation
st.header("Correlation and Causation")

correlation_features = ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight']
for feature in correlation_features:
    pearson_coef, p_value = stats.pearsonr(df[feature], df['price'])
    st.write(f"Feature: {feature}")
    st.write(f"- Pearson Correlation Coefficient: {pearson_coef}")
    st.write(f"- P-value: {p_value}")
    st.write("---")
