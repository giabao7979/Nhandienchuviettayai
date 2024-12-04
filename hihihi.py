import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageTk
import tensorflow as tf
import numpy as np
import cv2

# Tải mô hình đã huấn luyện
model = tf.keras.models.load_model('digit_model.h5')

# Kích thước canvas vẽ
width, height = 200, 200

# Đường dẫn ảnh nền
background_image_path = "E:/test2/anhnen.PNG"

# Danh sách chứa các dữ liệu mới và nhãn
new_data = []
new_labels = []

# Biến lưu kết quả nhận diện hiện tại
predicted_digit = None

# Hàm nhận diện chữ số
def recognize_digit():
    global predicted_digit
    img_resized = image1.resize((28, 28))
    img_resized = np.array(img_resized.convert('L')) / 255.0  # Chuẩn hóa ảnh
    img_resized = np.expand_dims(img_resized, axis=0)  # Thêm batch size
    prediction = model.predict(img_resized)
    predicted_digit = np.argmax(prediction)
    result_label.config(text=f"Số nhận diện được là: {predicted_digit}")

# Hàm thêm vào dữ liệu nếu kết quả đúng
def submit_correct():
    if predicted_digit is not None:
        img_resized = np.array(image1.resize((28, 28)).convert('L')) / 255.0
        new_data.append(img_resized)
        new_labels.append(predicted_digit)
        messagebox.showinfo("Thông báo", f"Đã lưu số {predicted_digit} vào dữ liệu huấn luyện!")
    else:
        messagebox.showerror("Lỗi", "Hãy nhận diện một số trước khi xác nhận!")

# Hàm thêm dữ liệu với nhãn do người dùng nhập
def submit_with_label():
    label = label_entry.get()
    if label.isdigit() and 0 <= int(label) <= 9:
        img_resized = np.array(image1.resize((28, 28)).convert('L')) / 255.0
        new_data.append(img_resized)
        new_labels.append(int(label))
        messagebox.showinfo("Thông báo", f"Đã lưu số {label} vào dữ liệu huấn luyện!")
    else:
        messagebox.showerror("Lỗi", "Vui lòng nhập một số hợp lệ từ 0 đến 9!")

# Hàm vẽ lên canvas
def paint(event):
    x1, y1 = (event.x - 3), (event.y - 3)
    x2, y2 = (event.x + 3), (event.y + 3)
    cv.create_line(x1, y1, x2, y2, fill="white", width=15)
    draw.line([x1, y1, x2, y2], fill="white", width=15)

# Hàm xóa canvas
def clear_canvas():
    cv.delete("all")
    global image1, draw, predicted_digit
    image1 = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image1)
    predicted_digit = None
    result_label.config(text="Số nhận diện được là: ")

# Hàm mở ảnh từ máy tính
def upload_image():
    filepath = filedialog.askopenfilename()
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28, 28)) / 255.0
    prediction = model.predict(np.expand_dims(img_resized, axis=0))
    predicted_digit = np.argmax(prediction)
    result_label.config(text=f"Số nhận diện được là: {predicted_digit}")

# Hàm hiển thị ảnh nền
def set_background():
    bg_image = Image.open(background_image_path)
    bg_image = bg_image.resize((root.winfo_width(), root.winfo_height()))  # Thay đổi kích thước khớp với cửa sổ
    bg_image_tk = ImageTk.PhotoImage(bg_image)
    background_label.config(image=bg_image_tk)
    background_label.image = bg_image_tk  # Lưu tham chiếu để tránh bị xóa bộ nhớ

# Giao diện người dùng với Tkinter
root = tk.Tk()
root.geometry("1280x960")
root.title("Handwritten Digit Recognition")

# Ảnh nền
background_label = tk.Label(root)
background_label.place(relwidth=1, relheight=1)  # Đặt ảnh nền toàn màn hình

# Canvas vẽ
cv = tk.Canvas(root, width=width, height=height, bg='black')
cv.bind("<B1-Motion>", paint)
cv.place(relx=0.5, rely=0.4, anchor="center")

# Nút xóa canvas
clear_button = tk.Button(root, text="Xóa", font=("Arial", 14), command=clear_canvas)
clear_button.place(relx=0.3, rely=0.5, anchor="center")

# Nút nhận diện chữ số
recognize_button = tk.Button(root, text="Nhận diện", font=("Arial", 14), command=recognize_digit)
recognize_button.place(relx=0.7, rely=0.5, anchor="center")

# Nút tải ảnh
upload_button = tk.Button(root, text="Tải hình ảnh", font=("Arial", 14), command=upload_image)
upload_button.place(relx=0.5, rely=0.55, anchor="center")

# Nhãn kết quả nhận diện
result_label = tk.Label(root, text="Số nhận diện được là: ", font=("Arial", 14), bg="white", fg="black")
result_label.place(relx=0.5, rely=0.6, anchor="center")

# Nhập nhãn tùy chỉnh
label_entry = tk.Entry(root, font=("Arial", 14))
label_entry.place(relx=0.5, rely=0.7, anchor="center")

# Nút thêm dữ liệu đúng
submit_correct_button = tk.Button(root, text="Xác nhận đúng", font=("Arial", 14), command=submit_correct)
submit_correct_button.place(relx=0.4, rely=0.8, anchor="center")

# Nút thêm dữ liệu với nhãn
submit_label_button = tk.Button(root, text="Thêm số đúng", font=("Arial", 14), command=submit_with_label)
submit_label_button.place(relx=0.6, rely=0.8, anchor="center")

# Khởi tạo ảnh ban đầu
image1 = Image.new("RGB", (width, height), "black")
draw = ImageDraw.Draw(image1)

# Đặt ảnh nền ban đầu
root.update_idletasks()
set_background()

root.mainloop()
