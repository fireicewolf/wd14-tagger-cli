import json
import os
import shutil
import requests

from pathlib import Path
from tqdm import tqdm
from typing import Union, Optional

from utils.logger import Logger


def download(
        logger: Logger,
        config_file: Path,
        model_name: str,
        model_site: str,
        models_save_path: Path,
        use_sdk_cache: bool = False,
        download_method: str = "sdk",
        force_download: bool = False
) -> tuple[Path, Path]:
    if os.path.isfile(config_file):
        logger.info(f'Using config: {str(config_file)}')
    else:
        logger.error(f'{str(config_file)} NOT FOUND!')
        raise FileNotFoundError

    def read_json(config_file, model_name) -> dict[str]:
        with open(config_file, 'r', encoding='utf-8') as config_json:
            datas = json.load(config_json)
            if model_name not in datas.keys():
                logger.error(f'"{str(model_name)}" NOT FOUND IN CONFIG!')
                raise FileNotFoundError
            return datas[model_name]

    model_info = read_json(config_file, model_name)

    models_save_path = Path(os.path.join(models_save_path, model_name))

    if use_sdk_cache:
        logger.warning('use_sdk_cache ENABLED! download_method force to use "SDK" and models_save_path will be ignored')
        download_method = 'sdk'
    else:
        logger.info(f'Model and csv will be stored in {str(models_save_path)}.')

    def download_choice(
            model_info: dict[str],
            model_site: str,
            models_save_path: Path,
            download_method: str = "sdk",
            use_sdk_cache: bool = False,
            force_download: bool = False
    ):
        if download_method.lower() == 'sdk':
            if model_site == "huggingface":
                model_hf_info = model_info["huggingface"]
                try:
                    from huggingface_hub import hf_hub_download
                    repo_id = model_hf_info["repo_id"]

                    logger.info(f'Will download onnx model from Huggingface repo: {repo_id}')
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=model_hf_info["onnx"],
                        revision=model_hf_info["revision"],
                        local_dir=models_save_path if not use_sdk_cache else None,
                        # local_dir_use_symlinks=False if not use_sdk_cache else "auto",
                        # resume_download=True,
                        force_download=force_download
                    )

                    logger.info(f'Will download tags csv from Huggingface repo: {repo_id}')
                    tags_csv_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=model_hf_info["csv"],
                        revision=model_hf_info["revision"],
                        local_dir=models_save_path if not use_sdk_cache else None,
                        # local_dir_use_symlinks=False if not use_sdk_cache else "auto",
                        # resume_download=True,
                        force_download=force_download
                    )

                except:
                    logger.warning('huggingface_hub not installed or download via it failed, '
                                   'retrying with URL method to download...')
                    model_path, tags_csv_path = download_choice(
                        model_info,
                        model_site,
                        models_save_path,
                        use_sdk_cache=False,
                        download_method="url",
                        force_download=force_download
                    )
                    return model_path, tags_csv_path

            elif model_site == "modelscope":
                model_ms_info = model_info["modelscope"]
                try:
                    if force_download:
                        logger.warning('Modelscope api not support force download, '
                                       'trying to remove model path before download!')
                        shutil.rmtree(models_save_path)

                    from modelscope.hub.file_download import model_file_download
                    repo_id = model_ms_info["repo_id"]

                    logger.info(f'Will download onnx model from Modelscope repo: {repo_id}')
                    model_path = model_file_download(
                        model_id=repo_id,
                        file_path=model_ms_info["onnx"],
                        revision=model_ms_info["revision"],
                        local_dir=models_save_path if not use_sdk_cache else None,
                    )

                    logger.info(f'Will download onnx model from Modelscope repo: {repo_id}')
                    tags_csv_path = model_file_download(
                        model_id=repo_id,
                        file_path=model_ms_info["csv"],
                        revision=model_ms_info["revision"],
                        local_dir=models_save_path if not use_sdk_cache else None,
                    )
                except:
                    logger.warning('modelscope not installed or download via it failed, '
                                   'retrying with URL method to download...')
                    model_path, tags_csv_path = download_choice(
                        model_info,
                        model_site,
                        models_save_path,
                        use_sdk_cache=False,
                        download_method="url",
                        force_download=force_download
                    )
                    return model_path, tags_csv_path
            else:
                logger.error('Invalid model site!')
                raise ValueError

        else:
            onnx_url = model_info[model_site]["onnx_url"]
            csv_url = model_info[model_site]["csv_url"]

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
                        pbar = tqdm(total=total_size, initial=0, unit='B', unit_divisor=1024, unit_scale=True,
                                    dynamic_ncols=True,
                                    desc=desc)
                    else:
                        pbar = tqdm(initial=0, unit='B', unit_divisor=1024, unit_scale=True, dynamic_ncols=True,
                                    desc=desc)

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
                        print(
                            f'"{local_file}" already exist, but can\'t get its size from "{url}". '
                            f'Won\'t re-download it.')
                    elif os.path.getsize(local_file) == total_size:
                        print(f'"{local_file}" already exist, and its size match with "{url}".')
                    else:
                        print(
                            f'"{local_file}" already exist, but its size not match with "{url}"!\n'
                            f'Will re-download this file...')
                        download_progress()
                else:
                    download_progress()

                return Path(os.path.join(local_dir, filename))

            logger.info(f'Will download onnx model from url: {onnx_url}')
            model_path = url_download(
                url=onnx_url,
                local_dir=models_save_path,
                force_filename=model_info[model_site]["onnx"],
                force_download=force_download
            )
            logger.info(f'Will download tags csv from url: {csv_url}')
            tags_csv_path = url_download(
                url=csv_url,
                local_dir=models_save_path,
                force_filename=model_info[model_site]["csv"],
                force_download=force_download
            )

        return model_path, tags_csv_path

    model_path, tags_csv_path = download_choice(
        model_info=model_info,
        model_site=model_site,
        models_save_path=models_save_path,
        download_method=download_method,
        use_sdk_cache=use_sdk_cache,
        force_download=force_download
    )

    return Path(model_path), Path(tags_csv_path)
