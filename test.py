from backend.news_service import run_full_pipeline

if st.button("Fetch Latest News"):
    result = run_full_pipeline("technology")
    st.write(result)