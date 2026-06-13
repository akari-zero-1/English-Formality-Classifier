# AI Formality Checker

Đây là một dự án ứng dụng Trí tuệ Nhân tạo để phân loại sắc thái văn bản tiếng Anh. Ứng dụng đánh giá xem một câu văn tiếng Anh mang sắc thái **Trang trọng (Formal)** hay **Suồng sã (Informal)**. Giao diện người dùng được xây dựng trực quan bằng **Streamlit**.

## Các mô hình AI được hỗ trợ

Hệ thống cho phép bạn tùy chọn 1 trong 3 mô hình AI tùy theo nhu cầu:
1. **RoBERTa (Chậm - Độ chính xác cao nhất)**: Dựa trên kiến trúc Transformer mạnh mẽ, cho kết quả chính xác nhất.
2. **DistilBERT (Cân bằng Tốc độ & Chính xác)**: Phiên bản tối ưu hóa của BERT, cho tốc độ dự đoán nhanh hơn với độ chính xác xấp xỉ.
3. **FastText (Siêu tốc - Phù hợp CPU)**: Tốc độ suy luận cực nhanh, tốn ít tài nguyên, rất phù hợp khi chạy trên các thiết bị không có GPU. *(Yêu cầu Python <= 3.12)*.

## 🛠 Yêu cầu hệ thống

- **Python**: Phiên bản 3.8 trở lên.
  *Lưu ý: Thư viện `fasttext` hiện tại không hỗ trợ Python 3.13+. Nếu bạn muốn sử dụng mô hình FastText, hãy đảm bảo môi trường Python của bạn từ **3.12 trở xuống**. Nếu dùng Python 3.13+, ứng dụng vẫn chạy bình thường với RoBERTa và DistilBERT.*
- **Mô hình**: Các mô hình đã huấn luyện cần được đặt sẵn trong thư mục `models/` (bao gồm `best_roberta_formality`, `best_distilbert_formality`, và `best_fasttext_formality.bin`).

##  Hướng dẫn cài đặt và chạy ứng dụng

**Bước 1: Mở terminal tại thư mục gốc của dự án.**

**Bước 2: (Khuyến nghị) Tạo và kích hoạt môi trường ảo (virtual environment).**
```bash
python -m venv venv

# Kích hoạt trên Windows:
venv\Scripts\activate

# Kích hoạt trên macOS/Linux:
source venv/bin/activate
```

**Bước 3: Cài đặt các thư viện phụ thuộc.**
```bash
pip install -r requirements.txt
```

**Bước 4: Khởi chạy ứng dụng Streamlit.**
```bash
streamlit run app.py
```

Sau khi chạy lệnh trên, trình duyệt của bạn sẽ tự động mở trang web (thông thường tại địa chỉ `http://localhost:8501`). Bạn chỉ cần nhập một câu tiếng Anh vào hộp thoại và bấm **"🔍 AI Phân tích ngay!"** để xem kết quả đánh giá mức độ trang trọng của câu văn.

##  Cấu trúc dự án cơ bản

- `app.py`: File khởi chạy giao diện web chính bằng Streamlit.
- `src/predictor.py`: Chứa class `FormalityPredictor` với logic tải mô hình (`roberta`, `distilbert`, `fasttext`) và xử lý dữ liệu để đưa ra dự đoán.
- `models/`: Thư mục lưu trữ các file mô hình đã được fine-tune/huấn luyện trước.
- `requirements.txt`: Danh sách các package Python cần thiết để dự án hoạt động trơn tru.
