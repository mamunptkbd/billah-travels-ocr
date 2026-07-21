import streamlit as st
from PIL import Image
import fitz  # PyMuPDF
import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
import io

# পেজ কনফিগারেশন
st.set_page_config(page_title="BILLAH TRAVELS - Smart OCR System", layout="wide")

# OCR ইঞ্জিন লোড
@st.cache_resource
def load_ocr_engine():
    return RapidOCR()

st.title("✈️ BILLAH TRAVELS - Smart Document OCR System")
st.caption("স্ক্যান করা ছবি বা পিডিএফ থেকে অটোমেটিক টেক্সট ও ফেস এক্সট্রাক্ট করুন")

st.write("---")

# সাইডবার
st.sidebar.header("📁 ফাইল আপলোড করুন")
uploaded_file = st.sidebar.file_uploader(
    "পিডিএফ (PDF) বা ইমেজ (JPG, PNG) নির্বাচন করুন", 
    type=["pdf", "png", "jpg", "jpeg"]
)

# ট্যাবসমূহ
tab1, tab2, tab3 = st.tabs(["📄 ফাইল প্রিভিউ", "🔍 OCR টেক্সট এক্সট্রাকশন", "📸 ফেস ও সিগনেচার"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1].lower()
    image_np = None

    if file_type in ["jpg", "jpeg", "png"]:
        pil_img = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(pil_img)
    elif file_type == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(pil_img)

    # ট্যাব ১: ফাইল প্রিভিউ
    with tab1:
        st.subheader("আপলোড করা ফাইলের প্রিভিউ")
        st.image(image_np, caption="আপলোড করা নথি", use_container_width=True)

    # ট্যাব ২: OCR টেক্সট এক্সট্রাকশন
    with tab2:
        st.subheader("নিখুঁত টেক্সট এক্সট্রাকশন")
        with st.spinner("লেখা পড়ার কাজ চলছে..."):
            engine = load_ocr_engine()
            result, _ = engine(image_np)
            
            extracted_texts = []
            if result:
                extracted_texts = [line[1] for line in result]
            
            if extracted_texts:
                st.success("✅ টেক্সট এক্সট্রাকশন সম্পন্ন হয়েছে!")
                
                full_text = "\n".join(extracted_texts)
                st.write("### 📝 সম্পূর্ণ টেক্সট একসাথে (Copy Text):")
                st.text_area("এখান থেকে কপি করুন:", value=full_text, height=150)
                
                st.write("---")
                st.write("### 🔍 লাইন বাই লাইন টেক্সট এডিটর:")
                for idx, text in enumerate(extracted_texts, 1):
                    st.text_input(f"লাইন {idx}", value=text, key=f"line_{idx}")
            else:
                st.warning("⚠️ কোনো টেক্সট খুঁজে পাওয়া যায়নি। পরিষ্কার ডকুমেন্ট ব্যবহার করুন।")

    # ট্যাব ৩: ফেস ডিটেকশন
    with tab3:
        st.subheader("ছবি ও সিগনেচার এক্সট্রাক্ট")
        st.write("ডকুমেন্ট থেকে মুখমণ্ডল (Face) ডিটেক্ট করা হচ্ছে...")
        
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            st.success(f"✅ {len(faces)} টি মুখমণ্ডল চিহ্নিত করা হয়েছে!")
            for i, (x, y, w, h) in enumerate(faces):
                face_img = image_np[y:y+h, x:x+w]
                st.image(face_img, caption=f"এক্সট্রাক্ট করা ছবি {i+1}", width=180)
        else:
            st.info("💡 অটো-ডিটেকশনে ছবি পাওয়া যায়নি।")

else:
    with tab1:
        st.info("👈 কাজ শুরু করতে বামদিকের সাইডবার থেকে যেকোনো একটি ফাইল আপলোড করুন।")