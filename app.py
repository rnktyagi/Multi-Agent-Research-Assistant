import streamlit as st
import requests

st.set_page_config(
    page_title="Multi-Agent Research Assistant",
    page_icon="🔍",
    layout="centered"
)

BACKEND_URL = "http://localhost:8000/research"

st.title("🔍 Multi-Agent Research Assistant")
st.markdown(
    "Enter a topic below. The multi-agent system will gather, synthesize, "
    "and generate a comprehensive executive report."
)

user_query = st.text_input(
    "Research Topic",
    placeholder="e.g., Multi-agent LLM architecture patterns...",
    label_visibility="collapsed"
)

if st.button("Start Research", type="primary", use_container_width=True):
    if not user_query.strip():
        st.warning("Please enter a research topic to begin.")
    else:
        with st.spinner("Agents are researching... This may take up to a minute."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    json={"query": user_query},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()

            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Ensure your FastAPI server is running on http://localhost:8000")
                st.stop()
            except requests.exceptions.Timeout:
                st.error("🚨 Request timed out. The research pipeline may be taking longer than expected.")
                st.stop()
            except requests.exceptions.HTTPError:
                # FastAPI returns a JSON body with a "detail" field on errors (e.g. our HTTPException)
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                st.error(f"🚨 Research pipeline failed: {detail}")
                st.stop()
            except Exception as e:
                st.error(f"🚨 An unexpected error occurred: {e}")
                st.stop()

        report = data.get("report", "No report was returned.")
        objectives = data.get("objectives", [])
        is_approved = data.get("is_approved", False)
        improvements = data.get("improvements", [])
        retries_used = data.get("retries_used", 0)

        st.success("Research Complete!")

        # Small status row so a viewer can see the agentic loop actually did something,
        # not just "here's some text" — useful both for debugging and for demoing the project.
        status_col1, status_col2, status_col3 = st.columns(3)
        status_col1.metric("Approved", "Yes" if is_approved else "No")
        status_col2.metric("Objectives", len(objectives))
        status_col3.metric("Retries Used", retries_used)

        if objectives:
            with st.expander("Research objectives covered"):
                for i, obj in enumerate(objectives, start=1):
                    st.markdown(f"{i}. {obj}")

        if improvements and not is_approved:
            with st.expander("Reviewer feedback (final round)"):
                for imp in improvements:
                    st.markdown(f"- {imp}")

        st.divider()
        st.markdown(report)