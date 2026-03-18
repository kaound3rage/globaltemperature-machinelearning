import kagglehub
import shutil
import os

# Download dataset (ke cache dulu)
path = kagglehub.dataset_download("thedevastator/global-land-and-surface-temperature-trends-analy")

# Folder tujuan = lokasi file Python ini
current_dir = os.path.dirname(os.path.abspath(__file__))

# Copy semua file ke folder script
for file_name in os.listdir(path):
    source = os.path.join(path, file_name)
    destination = os.path.join(current_dir, file_name)
    
    if os.path.isdir(source):
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)

print("Dataset berhasil dipindahkan ke:", current_dir)