import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from predict import predict_sales

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

st.set_page_config(
    page_title="Walmart Sales Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f5f7fb;
}

h1,h2,h3{
    color:#1f2937;
}

.metric-card{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 3px 10px rgba(0,0,0,0.1);
}

.sidebar .sidebar-content{
    background:#0E1117;
}

div.stButton > button{
    width:100%;
    background:#0F62FE;
    color:white;
    border-radius:8px;
    height:45px;
    font-size:18px;
}

div.stButton > button:hover{
    background:#0043CE;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------

@st.cache_data
def load_data():

    df = pd.read_csv("data/walmart_sales.csv")

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

    return df


df = load_data()

# ----------------------------------------------------
# LOAD MODEL RESULTS
# ----------------------------------------------------

@st.cache_data
def load_results():

    try:

        results = pd.read_csv("model/model_results.csv")

        # Convert index column into a normal column
        if "Unnamed: 0" in results.columns:
            results.rename(columns={"Unnamed: 0": "Model"}, inplace=True)

        # Rename R2 column to match the plotting code
        if "R2" in results.columns:
            results.rename(columns={"R2": "R2 Score"}, inplace=True)

        return results

    except:

        return None


results = load_results()

# ----------------------------------------------------
# SIDEBAR
# ----------------------------------------------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
    width=100
)

st.sidebar.title("Walmart Analytics")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 Dashboard",

        "🤖 Predict Sales",

        "📈 Model Performance",

        "ℹ️ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.write("Developed using")

st.sidebar.success("Python")

st.sidebar.success("Scikit-Learn")

st.sidebar.success("Streamlit")

st.sidebar.success("Plotly")

# ----------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------

def create_metric(label, value):

    st.markdown(f"""
    <div class="metric-card">
    <h3>{label}</h3>
    <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)


def format_currency(value):

    return f"${value:,.2f}"


def get_total_sales():

    return df["Weekly_Sales"].sum()


def get_average_sales():

    return df["Weekly_Sales"].mean()


def get_max_sale():

    return df["Weekly_Sales"].max()


def get_total_stores():

    return df["Store"].nunique()


# ----------------------------------------------------
# DATA PREPARATION
# ----------------------------------------------------

sales_over_time = (

    df.groupby("Date")["Weekly_Sales"]

    .sum()

    .reset_index()

)

store_sales = (

    df.groupby("Store")["Weekly_Sales"]

    .mean()

    .reset_index()

)

holiday_sales = (

    df.groupby("Holiday_Flag")["Weekly_Sales"]

    .mean()

    .reset_index()

)

monthly_sales = (

    df.assign(Month=df["Date"].dt.month)

      .groupby("Month")["Weekly_Sales"]

      .sum()

      .reset_index()

)

# ----------------------------------------------------
# HOME PAGE STARTS BELOW
# ----------------------------------------------------

if page == "🏠 Home":

    st.title("📈 Walmart Weekly Sales Prediction")

    st.write(
        """
        Welcome to the Walmart Weekly Sales Prediction Dashboard.

        This project predicts Walmart weekly sales using Machine Learning.

        It also provides an analytics dashboard to visualize historical sales,
        trends, and business insights.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Project Objective")

        st.write("""

Predict weekly Walmart sales using:

- Store

- Holiday Flag

- Temperature

- Fuel Price

- CPI

- Unemployment

- Year

- Month

- Week

        """)

    with col2:

        st.subheader("🛠 Technologies Used")

        st.write("""

✔ Python

✔ Pandas

✔ NumPy

✔ Scikit-Learn

✔ Plotly

✔ Streamlit

✔ Joblib

        """)

    st.markdown("---")

    st.subheader("Dataset Information")

    st.dataframe(df.head())

    st.info(f"Rows : {df.shape[0]}")

    st.info(f"Columns : {df.shape[1]}")

    st.info(f"Stores : {df['Store'].nunique()}")

    st.info(f"Years : {df['Date'].dt.year.nunique()}")
# ----------------------------------------------------
# DASHBOARD PAGE
# ----------------------------------------------------

elif page == "📊 Dashboard":

    st.title("📊 Walmart Sales Analytics Dashboard")

    st.markdown("### 📌 Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Total Sales",
            f"${get_total_sales():,.0f}"
        )

    with col2:
        st.metric(
            "📈 Average Weekly Sales",
            f"${get_average_sales():,.0f}"
        )

    with col3:
        st.metric(
            "🏪 Total Stores",
            get_total_stores()
        )

    with col4:
        st.metric(
            "🔥 Highest Weekly Sale",
            f"${get_max_sale():,.0f}"
        )

    st.markdown("---")

    ####################################################
    # SALES TREND
    ####################################################

    st.subheader("📈 Weekly Sales Trend")

    fig = px.line(
        sales_over_time,
        x="Date",
        y="Weekly_Sales",
        markers=True,
        title="Weekly Sales Over Time"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    ####################################################
    # STORE PERFORMANCE
    ####################################################

    st.markdown("---")

    st.subheader("🏪 Average Sales by Store")

    fig = px.bar(
        store_sales,
        x="Store",
        y="Weekly_Sales",
        color="Weekly_Sales",
        text_auto=".2s",
        title="Average Weekly Sales Per Store"
    )

    fig.update_layout(
        template="plotly_white",
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)

    ####################################################
    # HOLIDAY SALES
    ####################################################

    st.markdown("---")

    st.subheader("🎄 Holiday vs Non-Holiday Sales")

    holiday_sales["Holiday"] = holiday_sales["Holiday_Flag"].map({
        0: "Non-Holiday",
        1: "Holiday"
    })

    fig = px.pie(
        holiday_sales,
        names="Holiday",
        values="Weekly_Sales",
        hole=0.45,
        title="Average Weekly Sales"
    )

    st.plotly_chart(fig, use_container_width=True)

    ####################################################
    # MONTHLY SALES
    ####################################################

    st.markdown("---")

    st.subheader("📅 Monthly Sales")

    fig = px.bar(
        monthly_sales,
        x="Month",
        y="Weekly_Sales",
        color="Weekly_Sales",
        text_auto=".2s"
    )

    fig.update_layout(
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    ####################################################
    # NUMERICAL FEATURES
    ####################################################

    st.markdown("---")

    st.subheader("📊 Numerical Feature Distribution")

    feature = st.selectbox(

        "Select Feature",

        [

            "Temperature",

            "Fuel_Price",

            "CPI",

            "Unemployment",

            "Weekly_Sales"

        ]

    )

    fig = px.histogram(

        df,

        x=feature,

        nbins=30,

        marginal="box",

        title=f"{feature} Distribution"

    )

    st.plotly_chart(fig, use_container_width=True)

    ####################################################
    # SCATTER PLOT
    ####################################################

    st.markdown("---")

    st.subheader("📉 Feature Relationship")

    x_axis = st.selectbox(

        "X-axis",

        [

            "Temperature",

            "Fuel_Price",

            "CPI",

            "Unemployment"

        ],

        key="scatter_x"

    )

    fig = px.scatter(

        df,

        x=x_axis,

        y="Weekly_Sales",

        color="Holiday_Flag",

        hover_data=["Store"],

        title=f"{x_axis} vs Weekly Sales"

    )

    st.plotly_chart(fig, use_container_width=True)

    ####################################################
    # CORRELATION HEATMAP
    ####################################################

    st.markdown("---")

    st.subheader("🔥 Correlation Matrix")

    numeric_df = df.select_dtypes(include=np.number)

    corr = numeric_df.corr()

    fig = px.imshow(

        corr,

        text_auto=".2f",

        aspect="auto",

        color_continuous_scale="RdBu_r"

    )

    fig.update_layout(height=600)

    st.plotly_chart(fig, use_container_width=True)

    ####################################################
    # RAW DATA
    ####################################################

    st.markdown("---")

    st.subheader("📄 Dataset Preview")

    if st.checkbox("Show Dataset"):

        st.dataframe(df)

        st.success(f"Rows : {df.shape[0]}")

        st.success(f"Columns : {df.shape[1]}")
# ----------------------------------------------------
# PREDICTION PAGE
# ----------------------------------------------------

elif page == "🤖 Predict Sales":

    st.title("🤖 Walmart Weekly Sales Prediction")

    st.write(
        """
        Enter the required information below to predict
        Walmart Weekly Sales using the trained Machine Learning model.
        """
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    ####################################################
    # LEFT COLUMN
    ####################################################

    with col1:

        store = st.selectbox(
            "🏪 Store",
            sorted(df["Store"].unique())
        )

        holiday = st.selectbox(
            "🎄 Holiday Flag",
            [0, 1],
            format_func=lambda x: "Holiday" if x == 1 else "Non-Holiday"
        )

        temperature = st.number_input(
            "🌡 Temperature",
            min_value=-20.0,
            max_value=130.0,
            value=70.0
        )

        fuel_price = st.number_input(
            "⛽ Fuel Price",
            min_value=0.0,
            max_value=10.0,
            value=3.50
        )

    ####################################################
    # RIGHT COLUMN
    ####################################################

    with col2:

        cpi = st.number_input(
            "📊 CPI",
            min_value=100.0,
            max_value=300.0,
            value=210.0
        )

        unemployment = st.number_input(
            "👨‍💼 Unemployment",
            min_value=0.0,
            max_value=20.0,
            value=7.0
        )

        year = st.selectbox(
            "📅 Year",
            sorted(df["Date"].dt.year.unique())
        )

        month = st.selectbox(
            "📆 Month",
            list(range(1, 13))
        )

        week = st.slider(
            "📈 Week Number",
            1,
            52,
            25
        )

    st.markdown("---")

    ####################################################
    # INPUT SUMMARY
    ####################################################

    st.subheader("📋 Input Summary")

    summary = pd.DataFrame({

        "Feature": [

            "Store",
            "Holiday",
            "Temperature",
            "Fuel Price",
            "CPI",
            "Unemployment",
            "Year",
            "Month",
            "Week"

        ],

        "Value": [

            store,
            holiday,
            temperature,
            fuel_price,
            cpi,
            unemployment,
            year,
            month,
            week

        ]

    })

    st.dataframe(summary, use_container_width=True)

    st.markdown("---")

    ####################################################
    # PREDICT BUTTON
    ####################################################

    if st.button("🚀 Predict Weekly Sales"):

        try:

            prediction = predict_sales(

                store=store,

                holiday=holiday,

                temperature=temperature,

                fuel_price=fuel_price,

                cpi=cpi,

                unemployment=unemployment,

                year=year,

                month=month,

                week=week

            )

            st.success("Prediction Generated Successfully!")

            st.metric(

                "💰 Predicted Weekly Sales",

                f"${prediction:,.2f}"

            )

            ################################################

            # INTERPRETATION

            ################################################

            if prediction > df["Weekly_Sales"].mean():

                st.info(
                    "📈 Predicted sales are ABOVE the average weekly sales."
                )

            else:

                st.warning(
                    "📉 Predicted sales are BELOW the average weekly sales."
                )

            ################################################

            # COMPARISON

            ################################################

            st.subheader("📊 Prediction Comparison")

            comparison = pd.DataFrame({

                "Category": [

                    "Average Sales",

                    "Predicted Sales"

                ],

                "Sales": [

                    df["Weekly_Sales"].mean(),

                    prediction

                ]

            })

            fig = px.bar(

                comparison,

                x="Category",

                y="Sales",

                color="Category",

                text_auto=".2s"

            )

            fig.update_layout(

                template="plotly_white",

                showlegend=False

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        except Exception as e:

            st.error(

                "Prediction failed."

            )

            st.exception(e)

    ####################################################
    # SAMPLE INPUTS

    ####################################################

    st.markdown("---")

    with st.expander("📌 Example Input Values"):

        st.write("""

Store : 10

Holiday : Non-Holiday

Temperature : 72

Fuel Price : 3.50

CPI : 210

Unemployment : 7

Year : 2012

Month : 11

Week : 46

""")
# ----------------------------------------------------
# MODEL PERFORMANCE PAGE
# ----------------------------------------------------

elif page == "📈 Model Performance":

    st.title("📈 Model Performance")

    st.write(
        """
        Compare all machine learning models used for predicting
        Walmart Weekly Sales.
        """
    )

    st.markdown("---")

    # Load model results
    if results is not None:

        st.subheader("📋 Model Comparison")

        st.dataframe(results, use_container_width=True)

        st.markdown("---")

        ###################################################
        # R² SCORE
        ###################################################

        if "Model" in results.columns and "R2 Score" in results.columns:

            fig = px.bar(
                results,
                x="Model",
                y="R2 Score",
                color="R2 Score",
                text_auto=".3f",
                title="R² Score Comparison"
            )

            fig.update_layout(template="plotly_white")

            st.plotly_chart(fig, use_container_width=True)

        ###################################################
        # RMSE
        ###################################################

        if "RMSE" in results.columns:

            fig = px.bar(
                results,
                x="Model",
                y="RMSE",
                color="RMSE",
                text_auto=".2s",
                title="RMSE Comparison"
            )

            fig.update_layout(template="plotly_white")

            st.plotly_chart(fig, use_container_width=True)

        ###################################################
        # MAE
        ###################################################

        if "MAE" in results.columns:

            fig = px.bar(
                results,
                x="Model",
                y="MAE",
                color="MAE",
                text_auto=".2s",
                title="MAE Comparison"
            )

            fig.update_layout(template="plotly_white")

            st.plotly_chart(fig, use_container_width=True)

    else:

        st.warning("Model results not found.")
        st.info("Run train.py first to generate model_results.csv")

# ----------------------------------------------------
# ABOUT PAGE
# ----------------------------------------------------

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.header("📌 Project Overview")

    st.write("""
This application predicts Walmart Weekly Sales using Machine Learning.

It allows users to:

- Predict weekly sales
- Analyze historical sales
- Compare store performance
- Study holiday impact
- Explore feature relationships
- Compare ML model performance
""")

    st.markdown("---")

    st.header("🧠 Machine Learning Workflow")

    workflow = [
        "1️⃣ Data Collection",
        "2️⃣ Data Cleaning",
        "3️⃣ Feature Engineering",
        "4️⃣ Exploratory Data Analysis",
        "5️⃣ Model Training",
        "6️⃣ Model Evaluation",
        "7️⃣ Prediction",
        "8️⃣ Streamlit Deployment"
    ]

    for step in workflow:
        st.write(step)

    st.markdown("---")

    st.header("🛠 Technologies Used")

    tech = pd.DataFrame({

        "Technology":[
            "Python",
            "Pandas",
            "NumPy",
            "Scikit-Learn",
            "Plotly",
            "Streamlit",
            "Joblib"
        ],

        "Purpose":[
            "Programming",
            "Data Analysis",
            "Numerical Computing",
            "Machine Learning",
            "Visualization",
            "Web Application",
            "Model Saving"
        ]

    })

    st.table(tech)

    st.markdown("---")

    st.header("📂 Dataset Features")

    st.write("""
- Store
- Date
- Weekly Sales
- Holiday Flag
- Temperature
- Fuel Price
- CPI
- Unemployment
""")

    st.markdown("---")

    st.header("🎯 Prediction Target")

    st.success("Weekly_Sales")

    st.markdown("---")

    st.header("👨‍💻 Developed By")

    st.info("""
Machine Learning Project

Walmart Weekly Sales Prediction

Built using Python, Streamlit and Scikit-Learn.
""")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;
                color:gray;
                font-size:15px;'>

    📈 Walmart Weekly Sales Prediction Dashboard

    Developed using ❤️ Python | Streamlit | Scikit-Learn | Plotly

    </div>
    """,
    unsafe_allow_html=True
)