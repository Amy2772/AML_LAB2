import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Boston Housing Regression",
    page_icon="🏠",
    layout="wide"
)


st.title("🏠 Boston Housing Price Prediction")
st.write(
    "Regularized Regression using Lasso and Ridge "
    "with Cross-Validation and Grid Search"
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel("boston_housing_clean.xlsx")

    return df


df = load_data()


# ============================================================
# PREPARE DATA
# ============================================================

X = df.drop("MEDV", axis=1)

y = df["MEDV"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# LINEAR REGRESSION
# ============================================================

linear = LinearRegression()

linear.fit(
    X_train_scaled,
    y_train
)

linear_pred = linear.predict(
    X_test_scaled
)


# ============================================================
# LASSO
# ============================================================

alpha_values = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
    10,
    100
]


lasso_grid = GridSearchCV(

    Lasso(max_iter=50000),

    {
        "alpha": alpha_values
    },

    cv=5,

    scoring="neg_mean_squared_error",

    n_jobs=-1
)


lasso_grid.fit(
    X_train_scaled,
    y_train
)


lasso = lasso_grid.best_estimator_

lasso_pred = lasso.predict(
    X_test_scaled
)


# ============================================================
# RIDGE
# ============================================================

ridge_grid = GridSearchCV(

    Ridge(),

    {
        "alpha": alpha_values
    },

    cv=5,

    scoring="neg_mean_squared_error",

    n_jobs=-1
)


ridge_grid.fit(
    X_train_scaled,
    y_train
)


ridge = ridge_grid.best_estimator_

ridge_pred = ridge.predict(
    X_test_scaled
)


# ============================================================
# POLYNOMIAL RIDGE
# SELF-LEARNING COMPONENT
# ============================================================

poly_model = Pipeline([

    (
        "polynomial",
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "ridge",
        Ridge()
    )

])


poly_grid = GridSearchCV(

    poly_model,

    {
        "ridge__alpha": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    },

    cv=5,

    scoring="r2",

    n_jobs=-1
)


poly_grid.fit(
    X_train,
    y_train
)


poly_ridge = poly_grid.best_estimator_

poly_pred = poly_ridge.predict(
    X_test
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(y_true, prediction):

    mae = mean_absolute_error(
        y_true,
        prediction
    )

    mse = mean_squared_error(
        y_true,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        prediction
    )

    return mae, mse, rmse, r2


# Calculate metrics

linear_mae, linear_mse, linear_rmse, linear_r2 = calculate_metrics(
    y_test,
    linear_pred
)


lasso_mae, lasso_mse, lasso_rmse, lasso_r2 = calculate_metrics(
    y_test,
    lasso_pred
)


ridge_mae, ridge_mse, ridge_rmse, ridge_r2 = calculate_metrics(
    y_test,
    ridge_pred
)


poly_mae, poly_mse, poly_rmse, poly_r2 = calculate_metrics(
    y_test,
    poly_pred
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Prediction",
        "Model Comparison",
        "Dataset"
    ]
)


# ============================================================
# PAGE 1: PREDICTION
# ============================================================

if page == "Prediction":

    st.header("🏠 House Price Prediction")

    st.write(
        "Enter the house characteristics below."
    )


    col1, col2 = st.columns(2)


    with col1:

        CRIM = st.number_input(
            "CRIM",
            value=0.1
        )

        ZN = st.number_input(
            "ZN",
            value=0.0
        )

        INDUS = st.number_input(
            "INDUS",
            value=5.0
        )

        CHAS = st.number_input(
            "CHAS",
            value=0
        )

        NOX = st.number_input(
            "NOX",
            value=0.5
        )

        RM = st.number_input(
            "RM",
            value=6.0
        )

        AGE = st.number_input(
            "AGE",
            value=60.0
        )


    with col2:

        DIS = st.number_input(
            "DIS",
            value=4.0
        )

        RAD = st.number_input(
            "RAD",
            value=5.0
        )

        TAX = st.number_input(
            "TAX",
            value=300.0
        )

        PTRATIO = st.number_input(
            "PTRATIO",
            value=15.0
        )

        B = st.number_input(
            "B",
            value=390.0
        )

        LSTAT = st.number_input(
            "LSTAT",
            value=10.0
        )


    # Create input dataframe

    input_data = pd.DataFrame({

        "CRIM": [CRIM],

        "ZN": [ZN],

        "INDUS": [INDUS],

        "CHAS": [CHAS],

        "NOX": [NOX],

        "RM": [RM],

        "AGE": [AGE],

        "DIS": [DIS],

        "RAD": [RAD],

        "TAX": [TAX],

        "PTRATIO": [PTRATIO],

        "B": [B],

        "LSTAT": [LSTAT]

    })


    if st.button(
        "Predict House Price",
        type="primary"
    ):

        # Polynomial Ridge prediction

        prediction = poly_ridge.predict(
            input_data
        )[0]


        st.success(
            f"Predicted MEDV: {prediction:.2f}"
        )

        st.info(
            "The prediction is generated using "
            "Polynomial Ridge Regression."
        )


# ============================================================
# PAGE 2: MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("📊 Model Comparison")


    comparison = pd.DataFrame({

        "Model": [

            "Linear Regression",

            "Lasso Regression",

            "Ridge Regression",

            "Polynomial Ridge"

        ],

        "MAE": [

            linear_mae,

            lasso_mae,

            ridge_mae,

            poly_mae

        ],

        "MSE": [

            linear_mse,

            lasso_mse,

            ridge_mse,

            poly_mse

        ],

        "RMSE": [

            linear_rmse,

            lasso_rmse,

            ridge_rmse,

            poly_rmse

        ],

        "R2 Score": [

            linear_r2,

            lasso_r2,

            ridge_r2,

            poly_r2

        ]

    })


    st.dataframe(
        comparison.round(4),
        use_container_width=True
    )


    # Metrics

    st.subheader("Best Results")


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Polynomial Ridge R²",
            f"{poly_r2:.4f}"
        )


    with col2:

        st.metric(
            "Polynomial Ridge RMSE",
            f"{poly_rmse:.4f}"
        )


    with col3:

        st.metric(
            "Best Ridge Alpha",
            ridge_grid.best_params_["alpha"]
        )


    # Alpha information

    st.subheader("Best Alpha Values")


    st.write(
        "Best Lasso Alpha:",
        lasso_grid.best_params_["alpha"]
    )


    st.write(
        "Best Ridge Alpha:",
        ridge_grid.best_params_["alpha"]
    )


    st.write(
        "Best Polynomial Ridge Alpha:",
        poly_grid.best_params_["ridge__alpha"]
    )


    # Bar chart

    st.subheader("R² Score Comparison")

    chart_data = comparison.set_index(
        "Model"
    )["R2 Score"]

    st.bar_chart(
        chart_data
    )


# ============================================================
# PAGE 3: DATASET
# ============================================================

elif page == "Dataset":

    st.header("📋 Dataset")


    st.write(
        "Boston Housing Dataset"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Rows",
            df.shape[0]
        )


    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )


    with col3:

        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )


    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe().round(2),
        use_container_width=True
    )


    st.subheader("Features")

    st.write(
        list(X.columns)
    )


    st.subheader("Target Variable")

    st.write(
        "MEDV"
    )