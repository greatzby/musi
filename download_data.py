import gdown
import zipfile
import os
import shutil

ZIP_NAME = "musique_v1.0.zip"
FILE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"

# 下载
print("Downloading...")
gdown.download(id=FILE_ID, output=ZIP_NAME, quiet=False)

# 解压
print("Extracting...")
with zipfile.ZipFile(ZIP_NAME, "r") as z:
    z.extractall(".")

# 清理
os.remove(ZIP_NAME)
if os.path.exists("__MACOSX"):
    shutil.rmtree("__MACOSX")

print("Done. Check ./data/")