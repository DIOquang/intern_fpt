# Real-ESRGAN & CodeFormer App

Dự án này là một ứng dụng giao diện web (dùng Gradio) kết hợp hai công cụ AI mạnh mẽ:
- **Real-ESRGAN**: Dùng để nâng cao chất lượng ảnh và video (Upscale).
- **CodeFormer**: Dùng để nhận diện và phục hồi/làm nét khuôn mặt (Face Restoration).

## Cấu Trúc Dự Án

Dự án được tổ chức thành 2 module riêng biệt để dễ dàng quản lý mã nguồn:

- `real-esrgan/`: Chứa mã nguồn cho mô hình Real-ESRGAN. Thư mục này bao gồm file `infer.py` thực hiện logic nâng cấp hình ảnh.
- `codeformer/`: Chứa mã nguồn cho mô hình CodeFormer (sử dụng thư viện RetinaFace để nhận diện khuôn mặt). Thư mục này bao gồm file `face_infer.py`.
- `notebooks/`: Chứa file `.ipynb` dùng cho thử nghiệm và nghiên cứu thuật toán.

## Cài Đặt và Sử Dụng

1. Cài đặt các thư viện phụ thuộc:
```bash
pip install -r requirements.txt
```

2. Khởi chạy ứng dụng:
```bash
python app.py
```
Ứng dụng sẽ khả dụng ở cổng `http://127.0.0.1:7860/`. Mở trình duyệt và trải nghiệm giao diện người dùng để tải video/ảnh lên.

> **Lưu ý**: Lần đầu chạy tính năng khuôn mặt, hệ thống có thể mất chút thời gian để tự động tải các model trọng số (weights) của CodeFormer.