import os
import requests
import zipfile

def download_file(url, dest_path):
    """
    Download a file from a URL to dest_path using requests.
    Skips download if file already exists.
    """
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping download.")
        return
    print(f"Downloading from {url} to {dest_path} ...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print('File downloaded successfully')
        print(f"Downloaded {len(response.content) // 1024} Kb")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")

def unzip_file(zip_path, extract_to):
    """
    Unzip a .zip file to the specified folder.
    Skips extraction if folder is not empty.
    """
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f'{extract_to} already exists and is not empty. Skipping extraction.')
        return
    print(f"Unzipping {zip_path} to {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print("Unzip complete.")

def download_and_unzip(url, zip_path, extract_to):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    download_file(url, zip_path)
    unzip_file(zip_path, extract_to)
