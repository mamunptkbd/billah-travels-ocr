import fitz  # PyMuPDF
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import streamlit as st

# RapidOCR ইনিশিয়ালাইজেশন
engine = RapidOCR()


def extract_and_merge_form_lines(img_np):
    """OCR থেকে পাওয়া টেক্সটগুলোকে সঠিক ধারাবাহিকতায় সাজিয়ে

    এবং একই লাইনের শব্দগুলোকে একসাথে যুক্ত করে স্ট্রিম তৈরি করার ফাংশন।
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

    # ১. Y-center অনুযায়ী প্রাথমিকভাবে সর্ট
    items.sort(key=lambda item: item["y_center"])

    # ২. পাশাপাশি থাকা টেক্সটকে একই লাইনে গ্রুপ করা
    rows = []
    for item in items:
        placed = False
        for row in rows:
            avg_height = sum(i["height"] for i in row) / len(row)
            avg_y_center = sum(i["y_center"] for i in row) / len(row)

            # কাছাকাছি Y-পজিশনে থাকলে একই লাইনে রাখা হবে
            if abs(item["y_center"] - avg_y_center) <= (avg_height * 0.6):
                row.append(item)
                placed = True
                break

        if not placed:
            rows.append([item])

    # ৩. উপর থেকে নিচে লাইনগুলো সাজানো
    rows.sort(key=lambda r: sum(i["y_center"] for i in r) / len(r))

    # ৪. বাম থেকে ডানে টেক্সট জোড়া দিয়ে পূর্ণাঙ্গ প্যারাগ্রাফ তৈরি
    final_lines = []
    for row in rows:
        row.sort(key=lambda i: i["x_min"])
        line_str = "  ".join([i["text"] for i in row])
        final_lines.append(line_str)

    return "\n".join(final_lines)


# --- Streamlit UI ---
st.set_page_config(page_title="Document Text Extractor", layout="wide")
st.title("📄 ডকুমেন্ট টেক্সট এক্সট্রাক্টর")

uploaded_file = st.file_uploader("আপনার PDF ফাইলটি আপলোড করুন", type=["pdf"])

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    for page_num in range(len(doc)):
        page = doc[page_num]
        st.subheader(f"📖 পৃষ্ঠা {page_num + 1}")

        # PDF থেকে ইমেজ কনভার্সন
        pix = page.get_pixmap(dpi=200)
        img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n
        )

        # সঠিক সিকোয়েন্স অনুযায়ী টেক্সট এক্সট্রাক্ট করা
        full_extracted_text = extract_and_merge_form_lines(img_np)

        # একটি সিঙ্গেল টেক্সট বক্সে সম্পূর্ণ আউটপুট প্রদর্শন
        st.text_area(
            label="এক্সট্রাক্ট করা টেক্সট (অরিজিনাল বিন্যাস অনুযায়ী):",
            value=full_extracted_text,
            height=450,
            key=f"page_text_{page_num}",
        )
