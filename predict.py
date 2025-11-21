from ultralytics import YOLO
from PIL import Image
import os

# === 1. NẠP MÔ HÌNH YOLO ===
model_path = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject4\runs\detect\train4\weights\best.pt"
model = YOLO(model_path)

# === 2. ĐƯỜNG DẪN ẢNH GỐC ===
image_path = r"C:\Users\LUU VAN THANH HUY\PycharmProjects\PythonProject4\images_test\img_4.png"

# === 3. KIỂM TRA KÍCH THƯỚC ẢNH ===
img = Image.open(image_path)
print(f"Kích thước gốc của ảnh: {img.size}")  # (width, height)

# === 4. RESIZE ẢNH VỀ 640x640 (CÙNG KÍCH CỠ HUẤN LUYỆN) ===
target_size = (640, 640)
if img.size != target_size:
    print(f"Đang resize ảnh từ {img.size} về {target_size}...")
    img_resized = img.resize(target_size)
    resized_path = os.path.splitext(image_path)[0] + "_resized.png"
    img_resized.save(resized_path)
    print(f" Ảnh đã resize lưu tại: {resized_path}")
else:
    resized_path = image_path
    print(" Ảnh đã đúng kích thước 640x640, không cần resize.")

# === 5. DỰ ĐOÁN ===
print("\n Đang dự đoán biển số...")
results = model.predict(
    source=resized_path,
    conf=0.1,          # Độ tin cậy tối thiểu
    imgsz=640,         # Kích thước ảnh đầu vào
    save=True,         # Lưu ảnh có bbox
    show=False
)

# === 6. HIỂN THỊ THÔNG TIN KẾT QUẢ ===
for r in results:
    boxes = r.boxes
    print("\nKết quả nhận dạng (bounding boxes):")
    print(boxes)

    im_array = r.plot()  # vẽ bounding box
    im = Image.fromarray(im_array[..., ::-1])  # BGR → RGB
    result_path = os.path.join("detected_plates", "detected_result_1.jpg")
    os.makedirs("detected_plates", exist_ok=True)
    im.save(result_path)
    print(f" Ảnh có bounding box đã lưu tại: {result_path}")

print("\n Hoàn tất dự đoán.")
