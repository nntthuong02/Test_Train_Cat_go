# CatGo - Local AI Training Pipeline

Dự án này cung cấp một quy trình (pipeline) để tự huấn luyện (self-play) AI chơi Go 5x5 trên máy cục bộ bằng PyTorch, sau đó xuất ra định dạng ONNX để tích hợp vào ứng dụng Android.

## 🚀 Hướng dẫn sử dụng

### Bước 1: Cài đặt môi trường
Bạn nên sử dụng môi trường ảo (virtual environment).
```bash
pip install -r requirements.txt
```

### Bước 2: Huấn luyện AI (Self-Play Training)
Chạy script `train_ai.py`. Script này sẽ:
1. Tự chơi (self-play) thông qua MCTS để tạo dữ liệu.
2. Dùng dữ liệu đó để huấn luyện Neural Network (`GoNet`).
3. Lưu model vào file `go5x5_model.pth`.

```bash
python train_ai.py
```
*Lưu ý: Bạn có thể chỉnh số lượng game và epoch trong hàm `train()` ở cuối file `train_ai.py`.*

### Bước 3: Xuất sang định dạng ONNX
Để dùng được trong Kotlin/Android, bạn cần chuyển đổi model `.pth` sang `.onnx`.
```bash
python export_onnx.py
```
Sau khi chạy, bạn sẽ nhận được file `go5x5_model.onnx`.

## 📂 Thành phần dự án
- `game_logic.py`: Logic bàn cờ Go và thuật toán tìm kiếm MCTS.
- `model.py`: Cấu trúc Neural Network (CNN with Policy & Value heads).
- `train_ai.py`: Script chính để thực hiện quy trình huấn luyện.
- `export_onnx.py`: Utility để xuất model.
- `requirements.txt`: Danh sách các thư viện cần thiết.
