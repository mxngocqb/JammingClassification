# JammingClassification

Mục tiêu: bộ công cụ để huấn luyện, kiểm thử và suy luận mô hình phát hiện nhiễu/giả mạo GNSS (GPS) từ phổ/tần số và phổ ảnh.

## Nội dung chính
- Tập lệnh huấn luyện: `train_model.py`, `train_cnn_svm.py`, `train.py` trong `BKDATASET/`.
- Suy luận / inference: `inference_single.py`, `realtime_inference.py` (trong `BKDATASET/`).
- Tiện ích: `read_spectrogram.py`, `create_summary_image.py`, `convert_to_coreml.py`.

## Yêu cầu
- Python 3.8+ (hoặc môi trường `venv`/`.venv`).
- Thư viện chính: `torch`, `torchvision`, `numpy`, `scipy`, `librosa`, `matplotlib`.
- Nếu cần chuyển mô hình lớn (file `.pth`), cân nhắc cài `git lfs` để lưu trữ các checkpoint lớn.

Ví dụ cài nhanh (tùy môi trường):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # nếu có
pip install torch torchvision numpy scipy librosa matplotlib
```

## Cấu trúc dữ liệu
- `BKDATASET/train/` và `BKDATASET/test/`: dữ liệu đã được tách theo label (ví dụ `AM/`, `Chirp/`, `FM/`, `Normal/`).
- `dataset_clean/`: bản làm sạch dữ liệu mẫu.
- Lưu ý: các file checkpoint lớn nằm trong `BKDATASET/TrainingModel/` và các thư mục nhị phân khác.

## .gitignore / Thư mục bị bỏ theo dõi
Tệp `.gitignore` trong repo đã bao gồm:
- `bladeRF/` — thiết bị/firmware chứa mã nguồn bladeRF (không cần push).
- `BKDATASET/` — thư mục dataset/checkpoints lớn (khuyến nghị không push toàn bộ vào Git; sử dụng lưu trữ bên ngoài hoặc `git lfs`).

Nếu remote đã từng track các file lớn, loại bỏ tracking bằng:
```bash
git rm -r --cached bladeRF BKDATASET || true
git add .gitignore
git commit -m "Stop tracking bladeRF/ and BKDATASET/"
```

## Chạy nhanh (Quick start)
- Kiểm tra file mẫu hoặc chạy inference một file:
```bash
python inference_single.py --input path/to/sample.wav
```
- Huấn luyện mẫu (ví dụ):
```bash
python train_model.py --data BKDATASET/train --epochs 30 --batch-size 32
```

## Thực tế/Realtime
- Có mã realtime trong `BKDATASET/realtime_inference.py` và các biến thể `realtime_inference_alexnet.py` / `realtime_inference_mobinet.py`.

## Gợi ý khi có file lớn
- Dùng `git lfs` cho các file `.pth` hoặc chuyển checkpoints ra storage (Google Drive, S3, etc.).

## Góp ý & Liên hệ
- Nếu muốn đóng góp, tạo Pull Request với mô tả rõ ràng.
- Liên hệ: chủ repo trên GitHub `mxngocqb`.

## Giấy phép
- Xem file `LICENSE` trong repo.

---
Ghi chú: README này cung cấp hướng dẫn sơ bộ; tôi có thể mở rộng thêm phần API, ví dụ đầu vào/đầu ra chi tiết hoặc hướng dẫn chuyển sang CoreML nếu bạn muốn.
