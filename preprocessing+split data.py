#%%
import cv2
import os

# Path utama ke folder Dataset
dataset_path = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset'

#%%
# Fungsi untuk menerapkan CLAHE pada citra grayscale
def apply_clahe(image):
    # Membuat objek CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Menerapkan CLAHE pada gambar grayscale
    clahe_image = clahe.apply(image)
    
    return clahe_image

#%%
# Fungsi untuk menyimpan gambar tanpa ICC profile (tetap menggunakan PNG)
def save_image(image, output_path):
    # Pastikan gambar hanya memiliki satu channel (grayscale)
    if len(image.shape) == 2:  # Cek apakah citra grayscale
        # Menyimpan gambar dalam format PNG (tanpa ICC profile)
        cv2.imwrite(output_path, image)
    else:
        print(f"Skipping {output_path}: Image is not grayscale.")

#%%
# Loop melalui setiap folder (kategori emosi)
for emotion_folder in os.listdir(dataset_path):
    emotion_folder_path = os.path.join(dataset_path, emotion_folder)
    
    # Periksa apakah path adalah sebuah folder
    if os.path.isdir(emotion_folder_path):
        # Membuat folder output untuk menyimpan gambar hasil CLAHE
        output_folder = os.path.join('Dataset_CLAHE', emotion_folder)
        os.makedirs(output_folder, exist_ok=True)
        
        # Loop melalui setiap file di dalam folder emosi
        for image_file in os.listdir(emotion_folder_path):
            image_path = os.path.join(emotion_folder_path, image_file)
            
            # Memeriksa apakah path adalah sebuah file (gambar)
            if os.path.isfile(image_path):
                # Membaca gambar dalam mode grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                # Memastikan gambar sudah grayscale dan berukuran 224x224
                if image is not None and image.shape == (224, 224):
                    # Menerapkan CLAHE pada gambar grayscale
                    clahe_image = apply_clahe(image)
                    
                    # Menyimpan hasil gambar ke folder output dalam format PNG
                    output_image_path = os.path.join(output_folder, image_file)
                    save_image(clahe_image, output_image_path)
                else:
                    print(f"Skipping {image_file}: Gambar tidak dalam format grayscale atau ukuran yang tidak sesuai.")
                    
print("Proses CLAHE selesai untuk semua folder.")

# %%
import os
import shutil
from sklearn.model_selection import train_test_split

# Path utama ke folder Dataset
dataset_path = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset'
train_folder = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset_train'
test_folder = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset_testing'

# %%
# Membuat folder untuk training dan testing
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Fungsi untuk membagi dataset menjadi train dan test
def split_data(dataset_path, train_folder, test_folder, test_size=0.2):
    # Loop untuk setiap kategori emosi dalam dataset
    for emotion_folder in os.listdir(dataset_path):
        emotion_folder_path = os.path.join(dataset_path, emotion_folder)
        
        # Periksa apakah path adalah folder
        if os.path.isdir(emotion_folder_path):
            # Buat folder untuk setiap kategori di train dan test
            emotion_train_folder = os.path.join(train_folder, emotion_folder)
            emotion_test_folder = os.path.join(test_folder, emotion_folder)
            os.makedirs(emotion_train_folder, exist_ok=True)
            os.makedirs(emotion_test_folder, exist_ok=True)
            
            # Ambil semua gambar dalam folder kategori
            image_files = [f for f in os.listdir(emotion_folder_path) if os.path.isfile(os.path.join(emotion_folder_path, f))]
            
            # Split data: 80% untuk training, 20% untuk testing
            train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
            
            # Pindahkan file ke folder training dan testing
            for file in train_files:
                shutil.copy(os.path.join(emotion_folder_path, file), os.path.join(emotion_train_folder, file))
            
            for file in test_files:
                shutil.copy(os.path.join(emotion_folder_path, file), os.path.join(emotion_test_folder, file))
    
    print("Dataset berhasil dibagi menjadi training dan testing.")

# Memanggil fungsi untuk membagi dataset
split_data(dataset_path, train_folder, test_folder, test_size=0.2)

# %%
