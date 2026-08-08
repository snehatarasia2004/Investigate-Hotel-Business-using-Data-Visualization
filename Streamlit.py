"""
Hotel Bookings Dashboard
=========================
Run with:  streamlit run streamlit_app.py

Covers:
  - Booking share by hotel type
  - Monthly booking seasonality
  - Cancellation rate by hotel type
  - Cancellation rate vs. length of stay
  - Cancellation rate vs. lead time
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Hotel Bookings Dashboard",
    page_icon="🏨",
    layout="wide",
)

COLORS = {"City Hotel": "#2E86AB", "Resort Hotel": "#E67E22"}
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]


# -----------------------------------------------------------------
# DATA LOADING (cached so it only runs once per session)
# -----------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df["arrival_date_month"] = pd.Categorical(
        df["arrival_date_month"], categories=MONTH_ORDER, ordered=True
    )
    df["total_stay"] = df["stays_in_weekend_nights"] + df["stays_in_weekdays_nights"]
    df["lead_time_bucket"] = pd.cut(
        df["lead_time"],
        bins=[-1, 7, 30, 60, 90, 180, 365, df["lead_time"].max()],
        labels=["0-7", "8-30", "31-60", "61-90", "91-180", "181-365", "365+"],
    )
    return df


st.title("🏨 Hotel Bookings Dashboard")
st.caption("City Hotel vs. Resort Hotel — bookings, seasonality, and cancellation drivers")

# File upload OR default path — makes it easy to demo without a hardcoded file
uploaded_file = st.sidebar.file_uploader("Upload bookings CSV", type="csv")
DEFAULT_PATH = "hotel_bookings_data.csv"

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data(DEFAULT_PATH)
        st.sidebar.info(f"Using default file: {DEFAULT_PATH}")
    except FileNotFoundError:
        st.warning("Upload a CSV in the sidebar to get started.")
        st.stop()

# -----------------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------------
st.sidebar.header("Filters")

hotel_options = df["hotel"].unique().tolist()
selected_hotels = st.sidebar.multiselect(
    "Hotel type", options=hotel_options, default=hotel_options
)

years = sorted(df["arrival_date_year"].unique().tolist())
selected_years = st.sidebar.multiselect(
    "Arrival year", options=years, default=years
)

cancel_filter = st.sidebar.radio(
    "Booking status", options=["All", "Cancelled only", "Not cancelled only"]
)

# Apply filters
fdf = df[df["hotel"].isin(selected_hotels) & df["arrival_date_year"].isin(selected_years)]
if cancel_filter == "Cancelled only":
    fdf = fdf[fdf["is_canceled"] == 1]
elif cancel_filter == "Not cancelled only":
    fdf = fdf[fdf["is_canceled"] == 0]

if fdf.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# -----------------------------------------------------------------
# KPI ROW
# -----------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Bookings", f"{len(fdf):,}")
col2.metric("Cancellation Rate", f"{fdf['is_canceled'].mean()*100:.1f}%")
col3.metric("Avg. Lead Time", f"{fdf['lead_time'].mean():.0f} days")
col4.metric("Avg. Stay Length", f"{fdf['total_stay'].mean():.1f} nights")

st.divider()

# -----------------------------------------------------------------
# SECTION 1: BOOKING SHARE + SEASONALITY
# -----------------------------------------------------------------
st.header("1. Booking Volume & Seasonality")

c1, c2 = st.columns([1, 2])

with c1:
    share = fdf["hotel"].value_counts().reset_index()
    share.columns = ["hotel", "count"]
    fig_pie = px.pie(
        share, names="hotel", values="count", color="hotel",
        color_discrete_map=COLORS, title="Share of Bookings by Hotel Type", hole=0.35,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    monthly = (
        fdf.groupby(["arrival_date_month", "hotel"], observed=True)
        .size().reset_index(name="bookings")
    )
    fig_month = px.line(
        monthly, x="arrival_date_month", y="bookings", color="hotel",
        color_discrete_map=COLORS, markers=True,
        title="Bookings per Month by Hotel Type",
        labels={"arrival_date_month": "Month", "bookings": "Number of Bookings"},
    )
    st.plotly_chart(fig_month, use_container_width=True)

st.divider()

# -----------------------------------------------------------------
# SECTION 2: CANCELLATION RATE BY HOTEL
# -----------------------------------------------------------------
st.header("2. Cancellation Rate by Hotel Type")

cancel_hotel = (fdf.groupby("hotel")["is_canceled"].mean() * 100).reset_index()
cancel_hotel.columns = ["hotel", "cancellation_rate"]

fig_cancel = px.bar(
    cancel_hotel, x="hotel", y="cancellation_rate", color="hotel",
    color_discrete_map=COLORS, text_auto=".1f",
    title="Cancellation Rate by Hotel Type",
    labels={"cancellation_rate": "Cancellation Rate (%)"},
)
fig_cancel.update_traces(texttemplate="%{y:.1f}%")
st.plotly_chart(fig_cancel, use_container_width=True)

st.divider()

# -----------------------------------------------------------------
# SECTION 3: STAY LENGTH vs CANCELLATION
# -----------------------------------------------------------------
st.header("3. Cancellation Rate vs. Length of Stay")

max_stay = st.slider("Cap length-of-stay axis at (nights)", 5, 30, 15)
fdf["stay_bucket"] = fdf["total_stay"].clip(upper=max_stay)

stay_cancel = (
    fdf.groupby(["stay_bucket", "hotel"], observed=True)["is_canceled"]
    .mean().mul(100).reset_index(name="cancellation_rate")
)

fig_stay = px.line(
    stay_cancel, x="stay_bucket", y="cancellation_rate", color="hotel",
    color_discrete_map=COLORS, markers=True,
    title="Cancellation Rate vs. Total Nights Stayed",
    labels={"stay_bucket": f"Total Nights (capped at {max_stay})",
            "cancellation_rate": "Cancellation Rate (%)"},
)
st.plotly_chart(fig_stay, use_container_width=True)

st.divider()

# -----------------------------------------------------------------
# SECTION 4: LEAD TIME vs CANCELLATION
# -----------------------------------------------------------------
st.header("4. Cancellation Rate vs. Lead Time")

lead_cancel = (
    fdf.groupby(["lead_time_bucket", "hotel"], observed=True)["is_canceled"]
    .mean().mul(100).reset_index(name="cancellation_rate")
)

fig_lead = px.bar(
    lead_cancel, x="lead_time_bucket", y="cancellation_rate", color="hotel",
    color_discrete_map=COLORS, barmode="group", text_auto=".1f",
    title="Cancellation Rate by Lead Time Bucket",
    labels={"lead_time_bucket": "Lead Time (days before arrival)",
            "cancellation_rate": "Cancellation Rate (%)"},
)
fig_lead.update_traces(texttemplate="%{y:.1f}%")
st.plotly_chart(fig_lead, use_container_width=True)

st.divider()

# -----------------------------------------------------------------
# RAW DATA (optional, collapsible)
# -----------------------------------------------------------------
with st.expander("View filtered raw data"):
    st.dataframe(fdf.head(500), use_container_width=True)
    st.caption(f"Showing first 500 of {len(fdf):,} filtered rows.")
