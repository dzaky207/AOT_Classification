import cv2
import os

# Buka video
video_path = "AOT_S3P2.mkv" # Ganti dengan path videomu
output_folder = "frames1"
os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Dapatkan FPS video
frame_interval = int(fps / 1) # Tentukan FPS yang ingin didapat
frame_count = 0
saved_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Simpan hanya setiap frame_interval
    if frame_count % frame_interval == 0:
        frame_name = os.path.join(output_folder, f"WIT_{saved_frames:04d}.jpg")
        cv2.imwrite(frame_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 50])  # Simpan dengan kualitas 50%
        saved_frames += 1

    frame_count += 1

cap.release()
print(f"Extraction complete! {saved_frames} frames saved in '{output_folder}'")
