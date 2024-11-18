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
dataset_path = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset_CLAHE'
train_folder = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset_train'
test_folder = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset_testing'

# %%
# Membuat folder untuk training dan testing
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Fungsi untuk membagi dataset menjadi train dan test
def split_data(dataset_path, train_folder, test_folder, test_size=0.2):
    total_images_before_split = 0
    total_train_images = 0
    total_test_images = 0
    
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
            num_images = len(image_files)
            total_images_before_split += num_images
            
            # Split data: 80% untuk training, 20% untuk testing
            train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
            
            # Hitung jumlah gambar di set training dan testing untuk kategori ini
            num_train = len(train_files)
            num_test = len(test_files)
            total_train_images += num_train
            total_test_images += num_test
            
            # Pindahkan file ke folder training dan testing
            for file in train_files:
                shutil.copy(os.path.join(emotion_folder_path, file), os.path.join(emotion_train_folder, file))
            
            for file in test_files:
                shutil.copy(os.path.join(emotion_folder_path, file), os.path.join(emotion_test_folder, file))
            
            # Print jumlah gambar sebelum dan setelah split untuk setiap kategori
            print(f"Kategori '{emotion_folder}': Total gambar sebelum split: {num_images}, "
                  f"Training: {num_train}, Testing: {num_test}")
    
    # Print total gambar sebelum dan setelah split
    print(f"\nTotal gambar di seluruh dataset sebelum split: {total_images_before_split}")
    print(f"Total gambar setelah split - Training: {total_train_images}, Testing: {total_test_images}")

# Memanggil fungsi untuk membagi dataset
split_data(dataset_path, train_folder, test_folder, test_size=0.2)

# %%
import os
import cv2
import numpy as np

def load_images_from_folder(folder_path):
    """Fungsi untuk membaca semua gambar dalam folder"""
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):  # Pastikan itu adalah file
            img = cv2.imread(filepath)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def apply_roberts(images):
    """Fungsi untuk menerapkan Roberts edge detection pada daftar gambar"""
    processed_images = []
    # Kernel Roberts untuk arah diagonal
    roberts_x = np.array([[1, 0],
                          [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1],
                          [-1, 0]], dtype=np.float32)

    for img in images:
        # Konversi ke grayscale jika gambar memiliki 3 channel (RGB)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # OpenCV default ke BGR
        else:
            gray = img

        # Terapkan filter Roberts
        edge_x = cv2.filter2D(gray, -1, roberts_x)  # Roberts di arah x
        edge_y = cv2.filter2D(gray, -1, roberts_y)  # Roberts di arah y

        # Pastikan hasilnya dalam format float32
        edge_x = edge_x.astype(np.float32)
        edge_y = edge_y.astype(np.float32)

        # Gabungkan hasil Roberts x dan y
        roberts_combined = cv2.magnitude(edge_x, edge_y)

        # Normalisasi hasil untuk memastikan nilai berada pada rentang [0, 255]
        roberts_normalized = cv2.normalize(roberts_combined, None, 0, 255, cv2.NORM_MINMAX)

        # Ubah ke format uint8 agar kompatibel dengan gambar
        roberts_uint8 = np.uint8(roberts_normalized)

        processed_images.append(roberts_uint8)

    return processed_images

def save_images_to_folder(images, filenames, output_folder):
    """Fungsi untuk menyimpan gambar ke folder output"""
    os.makedirs(output_folder, exist_ok=True)  # Buat folder jika belum ada
    for img, filename in zip(images, filenames):
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, img)

def process_all_folders(input_folder, output_folder):
    """Fungsi untuk memproses semua subfolder dalam folder input"""
    for root, dirs, files in os.walk(input_folder):
        # Hitung subpath relatif untuk menjaga struktur folder
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        
        # Proses hanya jika ada file dalam folder ini
        if files:
            print(f"Memproses folder: {root}")
            images, filenames = load_images_from_folder(root)
            processed_images = apply_roberts(images)
            save_images_to_folder(processed_images, filenames, output_subfolder)

# Path ke folder dataset asli
train_folder = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset_train'
test_folder = r'D:\SEMESTER 7\COMPUTER VISION\Github Comvis\Klasifikasi\Dataset_testing'

# Path ke folder hasil Roberts
roberts_output_folder = "dataset_roberts"
train_roberts_folder = os.path.join(roberts_output_folder, "dataset_train")
test_roberts_folder = os.path.join(roberts_output_folder, "dataset_testing")

# Proses semua folder di dalam dataset_train dan dataset_testing
process_all_folders(train_folder, train_roberts_folder)
process_all_folders(test_folder, test_roberts_folder)

print(f"Hasil Roberts untuk dataset_train disimpan di folder: {train_roberts_folder}")
print(f"Hasil Roberts untuk dataset_testing disimpan di folder: {test_roberts_folder}")

# %%
