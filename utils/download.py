import os
import requests
from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional


def url_download(
        url: str,
        local_dir: Union[str, Path],
        force_download: bool = False,
        force_filename: Optional[str] = None
) -> Path:
    # Download file via url by requests library
    filename = os.path.basename(url) if not force_filename else force_filename
    local_file = os.path.join(local_dir, filename)

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    def download_progress():
        desc = 'Downloading {}'.format(filename)

        if total_size > 0:
            pbar = tqdm(total=total_size, initial=0, unit='B', unit_divisor=1024, unit_scale=True, dynamic_ncols=True,
                        desc=desc)
        else:
            pbar = tqdm(initial=0, unit='B', unit_divisor=1024, unit_scale=True, dynamic_ncols=True, desc=desc)

        if not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        with open(local_file, 'ab') as download_file:
            for data in response.iter_content(chunk_size=1024):
                if data:
                    download_file.write(data)
                    pbar.update(len(data))
        pbar.close()

    if not force_download and os.path.isfile(local_file):
        if total_size == 0:
            print(f'"{local_file}" already exist, but can\'t get its size from "{url}". Won\'t redownload it.')
        elif os.path.getsize(local_file) == total_size:
            print(f'"{local_file}" already exist, and its size match with "{url}".')
        else:
            print(f'"{local_file}" already exist, but its size not match with "{url}"!\nWill redownload this file...')
            download_progress()
    else:
        download_progress()

    return Path(os.path.join(local_dir, filename))
