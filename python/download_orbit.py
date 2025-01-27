#!/usr/bin/env python3
import os
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

def download_file(url, folder_path):
    local_filename = os.path.join(folder_path, url.split('/')[-1])
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {local_filename}")

def get_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    links = []
    for link in soup.find_all('a', href=True):
        links.append(urljoin(url, link['href']))
    return links

def download_files_from_website(base_url, folder_path, file_extension='.DAT'):
    os.makedirs(folder_path, exist_ok=True)
    all_links = get_links(base_url)
    for link in all_links:
        if link.endswith(file_extension):
            download_file(link, folder_path)
        elif link.startswith(base_url):
            download_files_from_website(link, folder_path, file_extension)

if __name__ == "__main__":
    # URL of the website to download from
    base_url = "https://soho.nascom.nasa.gov/data/ancillary/orbit/predictive/2008/"

    # Folder path to save the downloaded files
    folder_path = "../data/orbit"

    # File extension to download (change to whatever you need)
    file_extension = '.DAT'

    download_files_from_website(base_url, folder_path, file_extension)