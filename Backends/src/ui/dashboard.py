import streamlit as st


def metric_card(title, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div style="color:#94A3B8;">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_dashboard():
    st.markdown(
        """
        <div class="hero-card">
            <h1>🏏 CricVision AI</h1>
            <h3>Live Bowling Analytics from Umpire View Camera</h3>
            <p style="color:#CBD5E1;font-size:18px;">
                Track ball path, estimate speed, detect pitch point, classify length,
                and save every delivery for cricket performance analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        metric_card("Total Sessions", "0", "No sessions yet")

    with col2:
        metric_card("Deliveries", "0", "Balls analysed")

    with col3:
        metric_card("Fastest Ball", "0 km/h", "Estimated speed")

    with col4:
        metric_card("Accuracy", "0%", "Line/length consistency")

    st.write("")

    st.subheader("Project Roadmap")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🎥 Live Camera</h3>
                <p>Umpire-view camera setup for real-time bowling analysis.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="feature-card">
                <h3>🧠 Ball Detection</h3>
                <p>YOLO-based cricket ball detection and tracking pipeline.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="feature-card">
                <h3>📊 Analytics</h3>
                <p>Speed, length, pitch map, deviation, and session history.</p>
            </div>
            """,
            unsafe_allow_html=True
        )