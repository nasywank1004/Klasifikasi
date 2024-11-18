import os
import cv2

#path split data
train_folder = r'D:\Klasifikasi-main\Klasifikasi-main\Dataset_train'
test_folder = r'D:\Klasifikasi-main\Klasifikasi-main\Dataset_testing'

# Fungsi untuk menerapkan Gaussian Filtering pada semua gambar dalam folder yang memiliki subfolder kategori
def apply_gaussian_filter(folder_path, output_folder_path, kernel_size=(5, 5)):
    # Loop melalui setiap kategori dalam folder (misalnya anger, happiness, dll.)
    for category_folder in os.listdir(folder_path):
        category_folder_path = os.path.join(folder_path, category_folder)

        # Memastikan kategori adalah folder (bukan file)
        if os.path.isdir(category_folder_path):
            # Membuat folder output untuk kategori yang sama jika belum ada
            category_output_folder = os.path.join(output_folder_path, category_folder)
            os.makedirs(category_output_folder, exist_ok=True)

            # Iterasi semua file gambar di dalam subfolder kategori
            for image_file in os.listdir(category_folder_path):
                image_path = os.path.join(category_folder_path, image_file)

                # Memastikan file adalah gambar
                if os.path.isfile(image_path):
                    # Membaca gambar
                    image = cv2.imread(image_path)

                    # Jika gambar berhasil dibaca
                    if image is not None:
                        # Menerapkan Gaussian Filtering
                        filtered_image = cv2.GaussianBlur(image, kernel_size, 0)

                        # Menyimpan gambar hasil filtering ke folder output
                        output_path = os.path.join(category_output_folder, image_file)
                        cv2.imwrite(output_path, filtered_image)
                    else:
                        print(f"Gambar {image_file} tidak bisa dibaca.")
                else:
                    print(f"{image_file} bukan file yang valid.")

# Folder untuk menyimpan hasil filtering
training_output_folder = 'gaussian_train'
testing_output_folder = 'gaussian_test'

# Terapkan Gaussian Filtering pada folder *training* dan *testing*
apply_gaussian_filter(train_folder, training_output_folder)
apply_gaussian_filter(test_folder, testing_output_folder)