import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
import numpy as np
import matplotlib.pyplot as plt

# ================================
# **📂 Parameter Dataset**
# ================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_INITIAL = 10  # Training awal
EPOCHS_FINE_TUNE = 15 # Fine-tuning
DATASET_PATH = "dataset/"

# ================================
# **📌 Data Augmentation**
# ================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, 
    horizontal_flip=True,
    rotation_range=30,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=True
)

# ================================
# **📌 Load Model Pretrained (ResNet50)**
# ================================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layer awal

# **Tambahkan Custom Layer**
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
output_layer = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output_layer)

# ================================
# **📌 Compile & Training Awal**
# ================================
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("🔹 Training awal model (tanpa fine-tuning)...")
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS_INITIAL)

# ================================
# **🔥 Fine-Tuning (Unfreeze beberapa layer)**
# ================================
for layer in base_model.layers[-30:]:  # Buka 30 layer terakhir
    layer.trainable = True

# **Compile ulang dengan learning rate kecil**
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("🔥 Fine-tuning model...")
history_finetune = model.fit(train_data, validation_data=val_data, epochs=EPOCHS_FINE_TUNE)

# ================================
# **📌 Simpan Model**
# ================================
model.save("resnet50_anime_model.h5")  # Simpan model dalam format .h5
print("✅ Model berhasil disimpan!")

# ================================
# **📊 Evaluasi & Plot Hasil Training**
# ================================
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_finetune.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'] + history_finetune.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_finetune.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'] + history_finetune.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

print("✅ Training selesai!")

# ================================
# **🖼 Prediksi Gambar Baru**
# ================================
def predict_image(image_path):
    model = load_model("resnet50_anime_model.h5")  # Load model yang sudah disimpan
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    label = "WIT Studio" if prediction[0][0] < 0.5 else "MAPPA Studio"
    
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediksi: {label}")
    plt.show()

# Contoh prediksi
# predict_image("test_image.webp")
