import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
import streamlit as st
from streamlit_drawable_canvas import st_canvas

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


def process_interactive_canvas(pil_img, key_suffix):
    """ক্যানভাসে মাউস ড্র্যাগ সিলেকশন ও এক্সট্রাকশন প্রসেসিং ফাংশন"""
    col1, col2 = st.columns(2)

    w, h = pil_img.size
    # পেজটিকে ডিসপ্লেতে মানানসই সাইজে নিয়ে আসা
    max_width = 600
    scale = min(1.0, max_width / w)
    canvas_w = int(w * scale)
    canvas_h = int(h * scale)

    with col1:
        st.write(
            "👉 **ছবি বা পেজের উপর মাউস চেপে টেনে (Drag) অংশ সিলেক্ট করুন:**"
        )

        # ইন্টারেক্টিভ ক্যানভাস (শুরুতে সম্পূর্ণ খালি থাকবে)
        canvas_result = st_canvas(
            fill_color="rgba(0, 255, 0, 0.2)",  # ড্র্যাগ করলে হাল্কা সবুজ ব্যাকগ্রাউন্ড
            stroke_color="#00FF00",  # সবুজ বর্ডার
            stroke_width=2,
            background_image=pil_img,
            update_streamlit=True,
            width=canvas_w,
            height=canvas_h,
            drawing_mode="rect",  # চতুর্ভুজ ড্র্যাগ মোড
            key=f"canvas_{key_suffix}",
        )

    with col2:
        st.write("📝 **সিলেক্ট করা অংশের টেক্সট:**")
        extracted_text = ""

        # যদি ইউজার মাউস দিয়ে কোনো বক্স ড্র্যাগ করে থাকে
        if (
            canvas_result.json_data is not None
            and len(canvas_result.json_data["objects"]) > 0
        ):
            # সর্বশেষে ড্র্যাগ করা বক্সটির কোঅর্ডিনেট নেওয়া
            obj = canvas_result.json_data["objects"][-1]

            left = int(obj["left"] / scale)
            top = int(obj["top"] / scale)
            width = int(obj["width"] * obj["scaleX"] / scale)
            height = int(obj["height"] * obj["scaleY"] / scale)

            if width > 5 and height > 5:
                right = min(w, left + width)
                bottom = min(h, top + height)
                left = max(0, left)
                top = max(0, top)

                # শুধু সিলেক্ট করা অংশ ক্রপ করা
                cropped_pil = pil_img.crop((left, top, right, bottom))
                cropped_np = np.array(cropped_pil.convert("RGB"))

                # টেক্সট এক্সট্রাক্ট করা
                extracted_text = extract_and_merge_form_lines(cropped_np)
        else:
            extracted_text = "👈 বাম পাশের ছবিতে মাউস দিয়ে ড্র্যাগ করে কোনো অংশ সিলেক্ট করলে এখানে টেক্সট ভেসে উঠবে।"

        st.text_area(
            label="Extracted Text Area",
            value=extracted_text,
            height=450,
            key=f"text_{key_suffix}",
        )


# --- Streamlit UI Config ---
st.set_page_config(page_title="Drag & Extract OCR", layout="wide")
st.title("🖱️ মাউস ড্র্যাগ টেক্সট এক্সট্রাক্টর")

uploaded_file = st.file_uploader(
    "আপনার PDF বা Image ফাইল আপলোড করুন",
    type=["pdf", "png", "jpg", "jpeg"],
)

if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            page = doc[page_num]
            st.subheader(f"📖 পৃষ্ঠা {page_num + 1}")

            pix = page.get_pixmap(dpi=200)
            pil_img = Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples
            )

            process_interactive_canvas(pil_img, f"pdf_{page_num}")
    else:
        pil_img = Image.open(uploaded_file).convert("RGB")
        process_interactive_canvas(pil_img, "img")
