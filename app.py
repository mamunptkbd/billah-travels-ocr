import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import streamlit as st
from streamlit_cropper import st_cropper

# RapidOCR ইনিশিয়ালাইজেশন
engine = RapidOCR()


def extract_and_merge_form_lines(img_np):
    """OCR থেকে পাওয়া টেক্সটগুলোকে অরিজিনাল লেআউট অনুযায়ী

    সাজিয়ে একই লাইনে যুক্ত করার ফাংশন।
    """
    results, _ = engine(img_np)
    if not results:
        return ""

    items = []
    for box, text, conf in results:
        text_str = text.strip()
        if not text_str:
            continue

        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]

        y_min, y_max = min(ys), max(ys)
        x_min = min(xs)
        height = y_max - y_min
        y_center = (y_min + y_max) / 2.0

        items.append(
            {
                "text": text_str,
                "x_min": x_min,
                "y_center": y_center,
                "height": height,
            }
        )

    # ১. Y-center অনুযায়ী প্রাথমিক সর্টিং
    items.sort(key=lambda item: item["y_center"])

    # ২. একই লাইনের টেক্সটগুলোকে গ্রুপ করা
    rows = []
    for item in items:
        placed = False
        for row in rows:
            avg_height = sum(i["height"] for i in row) / len(row)
            avg_y_center = sum(i["y_center"] for i in row) / len(row)

            if abs(item["y_center"] - avg_y_center) <= (avg_height * 0.6):
                row.append(item)
                placed = True
                break

        if not placed:
            rows.append([item])

    # ৩. উপর থেকে নিচে লাইন সাজানো
    rows.sort(key=lambda r: sum(i["y_center"] for i in r) / len(r))

    # ৪. বাম থেকে ডানে জোড়া লাগানো
    final_lines = []
    for row in rows:
        row.sort(key=lambda i: i["x_min"])
        line_str = "  ".join([i["text"] for i in row])
        final_lines.append(line_str)

    return "\n".join(final_lines)


# --- Streamlit UI Config ---
st.set_page_config(
    page_title="Crop & Extract Document Text", layout="wide"
)
st.title("✂️ সিলেক্টিভ টেক্সট এক্সট্রাক্টর")

uploaded_file = st.file_uploader(
    "আপনার PDF বা Image ফাইল আপলোড করুন",
    type=["pdf", "png", "jpg", "jpeg"],
)

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    # ক. যদি PDF ফাইল হয়
    if file_type == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

        for page_num in range(len(doc)):
            page = doc[page_num]
            st.subheader(f"📖 পৃষ্ঠা {page_num + 1}")

            pix = page.get_pixmap(dpi=200)
            pil_img = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    "✂️ **বাম পাশের বর্ডার টেনে যতটুকু অংশ প্রয়োজন সিলেক্ট করুন:**"
                )
                # ইমেজের ওপর ক্রপ বক্স দেখাবে
                cropped_img = st_cropper(
                    pil_img,
                    realtime_update=True,
                    box_color="#00FF00",  # সবুজ রঙের সিলেকশন বক্স
                    aspect_ratio=None,
                    key=f"crop_pdf_{page_num}",
                )

            with col2:
                st.write("📝 **সিলেক্ট করা অংশের টেক্সট:**")
                if cropped_img:
                    cropped_np = np.array(cropped_img.convert("RGB"))
                    extracted_text = extract_and_merge_form_lines(cropped_np)

                    st.text_area(
                        label="Selected Region Text",
                        value=extracted_text,
                        height=400,
                        key=f"text_pdf_{page_num}",
                    )

    # খ. যদি সাধারণ ইমেজ ফাইল হয় (JPG / PNG)
    else:
        pil_img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.write(
                "✂️ **বাম পাশের বর্ডার টেনে যতটুকু অংশ প্রয়োজন সিলেক্ট করুন:**"
            )
            cropped_img = st_cropper(
                pil_img,
                realtime_update=True,
                box_color="#00FF00",
                aspect_ratio=None,
                key="crop_img",
            )

        with col2:
            st.write("📝 **সিলেক্ট করা অংশের টেক্সট:**")
            if cropped_img:
                cropped_np = np.array(cropped_img)
                extracted_text = extract_and_merge_form_lines(cropped_np)

                st.text_area(
                    label="Selected Region Text",
                    value=extracted_text,
                    height=400,
                    key="text_img",
                )
