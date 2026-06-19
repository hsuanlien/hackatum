import streamlit as st

st.set_page_config(
    page_title="MTU Safety Dashboard",
    layout="wide"
)

st.title("🏭 MTU Safety Inspection Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.metric("Workers Visible", 3)

with col2:
    st.metric("System Status", "Online")

st.divider()

st.subheader("Safety Monitoring")

st.success("Camera Active")
st.success("Tracking Active")
st.success("Worker Detection Active")

st.subheader("PPE Violations")

st.info("No violations detected")