import streamlit as st
from src.predictor import FormalityPredictor, is_fasttext_available

st.set_page_config(page_title="AI Formality Checker", page_icon="🤖")

st.title("🤖 Trợ lý AI Phân loại Sắc thái Văn bản")
st.markdown("Ứng dụng Trí tuệ Nhân tạo để đánh giá xem câu tiếng Anh mang sắc thái **Trang trọng (Formal)** hay **Suồng sã (Informal)**.")

# bộ chọn mô hình
st.write("### ⚙️ Cấu hình Hệ thống")
model_options = [
    "RoBERTa (Chậm - Độ chính xác cao nhất)",
    "DistilBERT (Cân bằng Tốc độ & Chính xác)",
]
if is_fasttext_available():
    model_options.append("FastText (Siêu tốc - Phù hợp CPU)")

model_choice = st.selectbox(
    "Vui lòng chọn Mô hình AI để dự đoán:",
    model_options
)

# Chuyển đổi chuỗi UI thành mã code để truyền vào Predictor
model_dict = {
    "RoBERTa (Chậm - Độ chính xác cao nhất)": "roberta",
    "DistilBERT (Cân bằng Tốc độ & Chính xác)": "distilbert",
    "FastText (Siêu tốc - Phù hợp CPU)": "fasttext"
}
selected_model_code = model_dict[model_choice]

if not is_fasttext_available():
    st.caption("FastText đang tắt trên Python 3.13+. Dùng Python 3.12 nếu bạn muốn bật mô hình này.")

# Nạp mô hình dựa trên lựa chọn, dùng cache
@st.cache_resource
def load_ai_engine(model_type):
    return FormalityPredictor(model_type=model_type)

with st.spinner(f'Đang nạp "Não bộ" {model_choice}...'):
    ai_engine = load_ai_engine(selected_model_code)

# dự đoán
st.write("---")
user_input = st.text_area("✍️ Hãy nhập một câu tiếng Anh vào đây:", 
                          "u should totally check out that new movie lol", 
                          height=100)

if st.button("🔍 AI Phân tích ngay!", type="primary"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập một câu văn trước khi phân tích.")
    else:
        label, score, top2_scores = ai_engine.predict(user_input)
        
        st.write("Kết quả từ AI:")
        if label == "Formal":
            st.success(f"**TRANG TRỌNG (FORMAL)**")
            st.info(f"Độ tự tin: **{score:.4f}%**")
        else:
            st.error(f"**SUỒNG SÃ (INFORMAL)**")
            st.warning(f"Độ tự tin: **{score:.4f}%**")

        st.write("Top-2 score:")
        st.write(f"1) {top2_scores[0][0]}: **{top2_scores[0][1]:.4f}%**")
        st.write(f"2) {top2_scores[1][0]}: **{top2_scores[1][1]:.4f}%**")