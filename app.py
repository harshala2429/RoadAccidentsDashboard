import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Road Accident Analysis", layout="wide")

st.title("🛣️ Road Accident Analysis — Dashboard")

DATA_PATH = Path("data/processed/accidents_clean.csv")
MODEL_PATH = Path("models/severity_model.joblib")
GEOJSON_PATH = Path("data/external/india_states.geojson")  # put your geojson here

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.warning(f"Processed data not found: {DATA_PATH}. Run preprocessing or upload a CSV.")
        return None
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

df = load_data()
model = load_model()

tab1, tab2, tab3 = st.tabs(["Exploration", "Map", "Predict Severity"])

with tab1:
    st.subheader("Explore")
    if df is None:
        uploaded = st.file_uploader("Upload a processed CSV with expected columns", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
    if df is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            state = st.selectbox("State", ["(All)"] + sorted([s for s in df["state"].dropna().unique().tolist()]), index=0)
        with col2:
            year = st.selectbox("Year", ["(All)"] + sorted([int(y) for y in df["year"].dropna().unique().tolist()]), index=0)
        with col3:
            vtype = st.selectbox("Vehicle Type", ["(All)"] + sorted([s for s in df["vehicle_type"].dropna().unique().tolist()]), index=0)

        dff = df.copy()
        if state != "(All)": dff = dff[dff["state"]==state]
        if year != "(All)": dff = dff[dff["year"]==year]
        if vtype != "(All)": dff = dff[dff["vehicle_type"]==vtype]

        st.metric("Rows", len(dff))
        st.bar_chart(dff.groupby("month")["state"].count())

        st.area_chart(dff.groupby("hour")["state"].count() if "hour" in dff.columns else dff.assign(hour=0).groupby("hour")["state"].count())

with tab2:
    st.subheader("Choropleth (State-level)")
    if df is not None and GEOJSON_PATH.exists():
        # Aggregate by state, year (optional)
        agg = df.groupby("state").size().reset_index(name="accidents")

        m = folium.Map(location=[22.9734, 78.6569], zoom_start=4)
        folium.Choropleth(
            geo_data=str(GEOJSON_PATH),
            data=agg,
            columns=["state", "accidents"],
            key_on="feature.properties.ST_NM", # adjust to your geojson prop
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Accident Count"
        ).add_to(m)
        st_folium(m, width=900, height=500)
    else:
        st.info("Provide `data/external/india_states.geojson` to see the choropleth.")

with tab3:
    st.subheader("Predict Accident Severity")
    if model is None:
        st.info("Train a model first (`python src/train_model.py`).")
    else:
        # Simple input form
        col1, col2, col3 = st.columns(3)
        with col1:
            istate = st.text_input("State", "maharashtra")
            iroad = st.selectbox("Road Type", ["highway","urban road","rural road"], index=0)
            iveh = st.text_input("Vehicle Type", "car")
        with col2:
            iwea = st.selectbox("Weather", ["clear","rain","fog","other"], index=0)
            iyear = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
            imonth = st.number_input("Month", min_value=1, max_value=12, value=6)
        with col3:
            iday = st.selectbox("Weekday", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], index=0)
            itod = st.selectbox("Time of Day", ["Night (0-5)","Morning (6-11)","Afternoon (12-17)","Evening (18-21)","Late (22-24)"], index=2)

        if st.button("Predict"):
            sample = pd.DataFrame([{
                "state": istate.lower(),
                "city": None,
                "vehicle_type": iveh.lower(),
                "weather": iwea.lower(),
                "road_type": iroad.lower(),
                "year": int(iyear),
                "month": int(imonth),
                "weekday": iday,
                "time_of_day": itod
            }])
            pred = model.predict(sample)[0]
            st.success(f"Predicted Severity: **{pred.title()}**")
