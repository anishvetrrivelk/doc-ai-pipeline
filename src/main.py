import streamlit as st
import json
from PIL import Image
import numpy as np

from src.parsers.text_extractor import extract_text_from_pdf
from src.parsers.table_extractor import extract_tables
from src.parsers.image_parser import extract_images
from src.parsers.equation_detector import detect_equations_in_text
from src.parsers.image_equation_detector import detect_equations_in_image
from src.utils.unified_output import build_page_json, build_document_json

st.set_page_config(page_title="Document AI Pipeline", layout="wide")
st.title("📄 Document AI Pipeline – Interactive Tester")

# ------------------- UPLOAD -------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_path = f"/tmp/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully!")

    # ------------------- MODULE TOGGLES -------------------
    st.sidebar.header("Enable/Disable Modules")
    run_text = st.sidebar.checkbox("Text Extraction", value=True)
    run_tables = st.sidebar.checkbox("Table Extraction", value=True)
    run_images = st.sidebar.checkbox("Image Extraction", value=True)
    run_eq_text = st.sidebar.checkbox("Text-based Equation Detection", value=True)
    run_eq_image = st.sidebar.checkbox("Image-based Equation Detection", value=True)

    # ------------------- PIPELINE -------------------
    pages_text = extract_text_from_pdf(pdf_path) if run_text else []
    tables = extract_tables(pdf_path) if run_tables else []
    images = extract_images(pdf_path) if run_images else []

    st.info("Building Unified JSON...")
    page_json_list = []

    for i, text in enumerate(pages_text, start=1):
        # Tables and images for this page
        page_tables = [t for t in tables if t["page"] == i]
        page_images = [img for img in images if img["page"] == i]

        # Equations
        eq_text = detect_equations_in_text(text) if run_eq_text else []
        eq_images = []
        if run_eq_image:
            for img in page_images:
                eqs_in_img = detect_equations_in_image(img["numpy_array"])
                eq_images.append({"image_bbox": img["bbox"], "equations": eqs_in_img})

        # Build page JSON
        page_json = build_page_json(
            page_num=i,
            text=text,
            tables=page_tables,
            images=page_images,
            equations_text=eq_text,
            equations_img=eq_images
        )
        page_json_list.append(page_json)

    full_doc_json = build_document_json(page_json_list)
    st.success("Pipeline Completed!")

    # ------------------- DISPLAY RESULTS -------------------
    st.header("📌 Extracted Text")
    for i, t in enumerate(pages_text, start=1):
        with st.expander(f"Page {i} Text"):
            st.write(t)

    st.header("📊 Extracted Tables")
    for t in tables:
        with st.expander(f"Page {t['page']} – Table {t['table_index']}"):
            st.dataframe(t["as_dataframe"])

    st.header("🖼 Extracted Images & Equations")
    for img in images:
        with st.expander(f"Page {img['page']} – Image"):
            st.image(img["numpy_array"], use_column_width=True)
            if run_eq_image:
                # Draw boxes around detected equations
                img_disp = img["numpy_array"].copy()
                eq_boxes = []
                for page_eq in full_doc_json["pages"]:
                    for img_eq in page_eq["images"]:
                        if img_eq["bbox"] == img["bbox"]:
                            eq_boxes = img_eq["equations"][0]["equations"] if img_eq["equations"] else []
                import cv2
                for b in eq_boxes:
                    cv2.rectangle(img_disp, (b["x"], b["y"]),
                                  (b["x"]+b["w"], b["y"]+b["h"]),
                                  color=(255,0,0), thickness=2)
                st.image(img_disp, caption="Equations highlighted", use_column_width=True)

    st.header("∑ Extracted Equations (Text)")
    st.write([eq for page in full_doc_json["pages"] for eq in page["equations"]])

    st.header("📦 Unified JSON Output")
    st.json(full_doc_json)

    st.download_button(
        label="Download JSON Output",
        data=json.dumps(full_doc_json, indent=4),
        file_name="doc_output.json",
        mime="application/json"
    )
