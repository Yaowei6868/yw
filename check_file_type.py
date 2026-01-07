import os
import struct

# 请确保您把那个文件重命名为了 unknown_data，并放在了当前目录下
FILE_PATH = "unknown_data"

def detect_file_type():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 找不到文件: {FILE_PATH}")
        print("请把那个下载的文件重命名为 'unknown_data' 并放在当前目录下！")
        return

    # 获取文件大小
    file_size = os.path.getsize(FILE_PATH)
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")

    if file_size < 0.1: # 小于 100KB
        print("⚠️ 警告：文件太小了！可能下载失败，只是一个 HTML 网页或链接文件。")
    
    # 读取前 10 个字节 (Magic Number)
    with open(FILE_PATH, 'rb') as f:
        header = f.read(10)
    
    hex_header = header.hex().upper()
    print(f"文件头 (Hex): {hex_header}")

    # 判断类型
    if hex_header.startswith("504B0304"):
        print("✅ 这是一个 ZIP 压缩包！")
        print("👉 解决办法：请把文件重命名为 'data.zip'，然后解压。")
    elif hex_header.startswith("52617221"):
        print("✅ 这是一个 RAR 压缩包！")
        print("👉 解决办法：请把文件重命名为 'data.rar'，然后解压。")
    elif hex_header.startswith("1F8B"):
        print("✅ 这是一个 GZIP 压缩包！")
        print("👉 解决办法：请把文件重命名为 'data.tar.gz'，然后解压。")
    elif hex_header.startswith("934E554D5059"):
        print("✅ 这是一个 NumPy (.npy) 文件！")
        print("👉 这可能只是特征文件，您可能还需要另外下载 edges 和 labels。")
    elif hex_header.startswith("8003") or hex_header.startswith("8004"):
        print("✅ 这是一个 Python Pickle (.pkl) 文件！")
        print("👉 这可能包含完整数据。")
    elif "3C21444F43" in hex_header or "3C68746D6C" in hex_header: # <!DOC or <html
        print("❌ 这是一个 HTML 网页！")
        print("👉 原因：您可能在 GitHub 页面上直接右键另存为了链接，而不是文件。请点击进入文件详情页再下载。")
    else:
        print("❓ 未知格式，请把上面的 '文件头 (Hex)' 发给我。")

if __name__ == "__main__":
    detect_file_type()