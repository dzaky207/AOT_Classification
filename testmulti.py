import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import csv

test_folder = "test_images/"
output_csv = "hasil_klasifikasi.csv"

model = tf.keras.models.load_model("resnet50_anime_model.h5")

# Fungsi untuk prediksi gambar
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize gambar
    img_array = img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

    prediction = model.predict(img_array)[0][0]  # Ambil hasil prediksi
    label = "MAPPA Studio" if prediction < 0.5 else "WIT Studio"

    return label

# Membuka file CSV untuk menulis hasil prediksi
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Nama Gambar", "Hasil Klasifikasi"])  # Header CSV

    for img_name in os.listdir(test_folder):
        img_path = os.path.join(test_folder, img_name)
        hasil = predict_image(img_path)
        writer.writerow([img_name, hasil])

print(f"Hasil klasifikasi disimpan dalam {output_csv}")
