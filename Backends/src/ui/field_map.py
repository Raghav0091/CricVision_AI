"""Field Setup Lab — developer-only advanced field configuration."""

import json

import streamlit as st

from Backends.src.analysis.field_zones import classify_shot_zone, get_active_field_setup, set_active_field_setup
from Backends.src.data.field_presets import PRESET_NAMES
from Backends.src.ui.interactive_field_map import (
    build_field_setup,
    render_interactive_field_map,
)
from Backends.src.ui.theme import render_page_header, render_section_title, render_status_pill


def show_field_map_page():
    render_page_header(
        "Field Setup Lab",
        "Test presets, drag fielders, and validate geometry. Normal workflow uses Analyze or Live Session.",
        badge="Advanced",
    )

    active = get_active_field_setup()
    col1, col2 = st.columns([0.9, 1.3], gap="large")

    with col1:
        render_section_title("Controls")
        handedness = st.selectbox(
            "Batter handedness",
            ["Right-handed", "Left-handed"],
            index=0 if not str(active.get("batter_handedness", "")).lower().startswith("left") else 1,
            key="field_lab_handedness",
        )
        preset_options = PRESET_NAMES + (["Custom"] if active.get("preset") not in PRESET_NAMES else [])
        preset_name = st.selectbox(
            "Field preset",
            preset_options,
            index=preset_options.index(active.get("preset", "Balanced"))
            if active.get("preset", "Balanced") in preset_options
            else 0,
            key="field_lab_preset",
        )
        show_labels = st.checkbox("Show labels", value=True, key="field_lab_show_labels")

        if st.button("Reset to preset", use_container_width=True, key="field_lab_reset"):
            active = build_field_setup(preset_name, handedness)
            st.session_state["current_field_setup"] = active
            st.rerun()

        if st.button("Save Field Setup", type="primary", use_container_width=True, key="field_lab_save"):
            set_active_field_setup(active)
            st.success("Field setup saved.")

        with st.expander("Shot Direction Preview", expanded=False):
            shot_angle = st.slider("Shot angle", 0, 359, 45, key="field_lab_shot_angle")
            zone = classify_shot_zone(shot_angle, handedness)
            st.markdown(f"{render_status_pill(zone, 'success')} {render_status_pill(f'{shot_angle}°', 'gold')}", unsafe_allow_html=True)

    with col2:
        render_section_title("Interactive Field Map")
        active = render_interactive_field_map(
            field_setup=active,
            handedness=handedness,
            preset_name=preset_name,
            editable=True,
            compact=False,
            show_labels=show_labels,
            key="field_lab_map",
            shot_angle=st.session_state.get("field_lab_shot_angle"),
            selected_zone=classify_shot_zone(st.session_state.get("field_lab_shot_angle", 45), handedness),
        )
        st.session_state["current_field_setup"] = active

        with st.expander("Advanced — raw angle/radius", expanded=False):
            st.dataframe(
                [
                    {
                        "name": item.get("name"),
                        "position": item.get("position"),
                        "angle": item.get("angle"),
                        "radius": item.get("radius"),
                    }
                    for item in active.get("fielders", [])
                ],
                use_container_width=True,
            )
            export_json = json.dumps(active, indent=2)
            st.download_button("Export field JSON", export_json, file_name="field_setup.json", use_container_width=True)
            imported = st.text_area("Import field JSON", key="field_lab_import_json")
            if st.button("Apply imported JSON", key="field_lab_apply_import"):
                try:
                    parsed = json.loads(imported)
                    st.session_state["current_field_setup"] = parsed
                    st.success("Imported field setup applied.")
                    st.rerun()
                except json.JSONDecodeError as error:
                    st.error(f"Invalid JSON: {error}")
