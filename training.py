import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Tải và chuẩn bị dữ liệu MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Chuẩn hóa dữ liệu (mặc định là giá trị từ 0-255, chuyển sang [0, 1])
x_train = x_train / 255.0
x_test = x_test / 255.0

# Xây dựng mô hình Neural Network với TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # Chuyển ảnh 28x28 thành vector 1 chiều
    tf.keras.layers.Dense(128, activation='relu'),  # Lớp ẩn với 128 neuron
    tf.keras.layers.Dropout(0.2),  # Dropout để tránh overfitting
    tf.keras.layers.Dense(10, activation='softmax')  # Lớp đầu ra, 10 lớp (0-9)
])

# Biên dịch mô hình
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Huấn luyện mô hình
model.fit(x_train, y_train, epochs=5)

# Đánh giá mô hình trên tập kiểm tra
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Lưu mô hình vào tệp digit_model.h5
model.save('digit_model.h5')

# In hình ảnh mẫu và kết quả dự đoán
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"True Label: {y_test[0]}")
plt.show()

# In kết quả dự đoán
prediction = model.predict(np.expand_dims(x_test[0], axis=0))  # Dự đoán với hình ảnh mẫu
predicted_label = np.argmax(prediction)
print(f"Predicted Label: {predicted_label}")
