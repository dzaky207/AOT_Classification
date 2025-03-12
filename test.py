import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load model yang sudah dilatih
model = tf.keras.models.load_model("resnet50_anime_model.h5")

# Fungsi untuk prediksi gambar
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize gambar
    img_array = img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

    prediction = model.predict(img_array)[0][0]  # Ambil hasil prediksi
    label = "WIT Studio" if prediction < 0.5 else "MAPPA Studio"

    # Tampilkan gambar dengan hasil prediksi
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediksi: {label}")
    plt.show()

    return label

# Contoh prediksi
hasil = predict_image("test1.png")
print("Studio Prediksi:", hasil)
