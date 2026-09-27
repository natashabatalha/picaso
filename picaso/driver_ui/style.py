import streamlit as st

# Widget testids to narrow. Tables (st.dataframe/st.data_editor -> "stDataFrame"),
# buttons, and other non-input elements are intentionally left untouched so
# they can stay full width.
_NARROW_WIDGET_TESTIDS = [
    "stTextInput",
    "stNumberInput",
    "stSelectbox",
    "stMultiSelect",
    "stDateInput",
    "stTextArea",
    "stSlider",
]


def inject_narrow_input_css(max_width="50%"):
    """
    Caps dropdown/text/number input widgets at `max_width` of their
    container instead of stretching to fill it, while leaving tables
    (st.dataframe/st.data_editor) and buttons full width.

    Call once near the top of every page script -- CSS injected on one
    page does not carry over to another in a multipage app.
    """
    selector = ", ".join(f'div[data-testid="{testid}"]' for testid in _NARROW_WIDGET_TESTIDS)
    st.markdown(
        f"<style>{selector} {{ max-width: {max_width}; }}</style>",
        unsafe_allow_html=True,
    )
