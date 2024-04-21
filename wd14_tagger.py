import argparse
import csv
import glob
import json
import numpy
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from onnxruntime.capi.onnxruntime_inference_collection import InferenceSession
from tqdm import tqdm

from utils.image import image_process
from utils.logger import Logger

SUPPORT_IMAGE_FORMATS = ("bmp", "jpg", "jpeg", "png")
kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


class Tagger:
    def __init__(self, logger: Logger):
        self.logger = logger

    def download(
            self,
            # config_file: Path,
            model_name: str,
            model_site: str,
            models_save_path: Path,
            use_sdk_cache: bool = False,
            download_method: str = "sdk",
            force_download: bool = False
    ) -> tuple[Path, Path]:

        config_file = os.path.join(Path(__file__).parent, 'config', 'default.json')
        if os.path.isfile(config_file):
            self.logger.info(f'Using config: {str(config_file)}')
        else:
            self.logger.error(f'{str(config_file)} NOT FOUND!')
            raise FileNotFoundError

        def read_json(config_file, model_name) -> tuple[str]:
            datas = {}
            with open(config_file, 'r', encoding='utf-8') as config_json:
                datas = json.load(config_json)
            return datas[model_name]

        model_info = read_json(config_file, model_name)
        models_save_path = Path(os.path.join(models_save_path, model_name))

        if use_sdk_cache:
            self.logger.warning(
                'use_sdk_cache ENABLED! download_method force to use "SDK" and models_save_path will be ignored')
            download_method = 'sdk'
        else:
            self.logger.info(f'Model and csv will be stored in {str(models_save_path)}.')

        def download_choice(
                model_info: dict[str],
                model_site: str,
                models_save_path: Path,
                download_method: str = "sdk",
                use_sdk_cache: bool = False,
                force_download: bool = False
        ):
            model_path, tags_csv_path = None, None
            if download_method.lower() == 'sdk':
                if model_site == "huggingface":
                    model_hf_info = model_info["huggingface"]
                    try:
                        from huggingface_hub import hf_hub_download
                        repo_id = model_hf_info["repo_id"]

                        self.logger.info(f'Will download onnx model from huggingface repo: {repo_id}')
                        model_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=model_hf_info["onnx"],
                            revision=model_hf_info["revision"],
                            local_dir=models_save_path if not use_sdk_cache else None,
                            local_dir_use_symlinks=False if not use_sdk_cache else "auto",
                            resume_download=True,
                            force_download=force_download
                        )

                        self.logger.info(f'Will download tags csv from huggingface repo: {repo_id}')
                        tags_csv_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=model_hf_info["csv"],
                            revision=model_hf_info["revision"],
                            local_dir=models_save_path if not use_sdk_cache else None,
                            local_dir_use_symlinks=False if not use_sdk_cache else "auto",
                            resume_download=True,
                            force_download=force_download
                        )

                    except:
                        self.logger.warning('huggingface_hub not installed or download via it failed, '
                                            'retrying with URL method to download...')
                        download_choice(model_info, model_site, models_save_path, use_sdk_cache=False,
                                        download_method="url")

                elif model_site == "modelscope":
                    model_ms_info = model_info["modelscope"]
                    try:
                        if force_download:
                            self.logger.warning(
                                'modelscope api not support force download, '
                                'trying to remove model path before download!')
                            shutil.rmtree(models_save_path)

                        from modelscope.hub.file_download import model_file_download
                        repo_id = model_ms_info["repo_id"]

                        self.logger.info(f'Will download onnx model from huggingface repo: {repo_id}')
                        model_path = model_file_download(
                            model_id=repo_id,
                            file_path=model_ms_info["onnx"],
                            revision=model_ms_info["revision"],
                            cache_dir=models_save_path if not use_sdk_cache else None,
                        )

                        self.logger.info(f'Will download onnx model from huggingface repo: {repo_id}')
                        tags_csv_path = model_file_download(
                            model_id=repo_id,
                            file_path=model_ms_info["csv"],
                            revision=model_ms_info["revision"],
                            cache_dir=models_save_path if not use_sdk_cache else None,
                        )
                    except:
                        self.logger.warning('modelscope not installed or download via it failed, '
                                            'retrying with URL method to download...')
                        download_choice(
                            model_info,
                            model_site,
                            models_save_path,
                            use_sdk_cache=False,
                            download_method="url"
                        )
                else:
                    self.logger.error('Invalid model site!')
                    raise ValueError

            else:
                onnx_url = model_info[model_site]["onnx_url"]
                csv_url = model_info[model_site]["csv_url"]

                from utils.download import url_download
                self.logger.info(f'Will download onnx model from url: {onnx_url}')
                model_path = url_download(
                    url=onnx_url,
                    local_dir=models_save_path,
                    force_filename=model_info[model_site]["onnx"],
                    force_download=force_download
                )
                self.logger.info(f'Will download tags csv from url: {csv_url}')
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

    def load_model(
            self,
            model_path: Path,
            force_use_cpu: bool = False,
    ) -> InferenceSession:
        if not os.path.exists(model_path):
            self.logger.error(f'{str(model_path)} NOT FOUND!')
            raise FileNotFoundError

        import onnx
        import onnxruntime as ort

        self.logger.info(f'Loading model from {str(model_path)}')
        model = onnx.load(model_path)

        # get model batch size info
        try:
            model_batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
        except:
            model_batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param

        if args.batch_size != model_batch_size and not isinstance(model_batch_size, str) and model_batch_size > 0:
            # some rebatch model may use 'N' as dynamic axes
            self.logger.warning(
                f'Batch size {args.batch_size} doesn\'t match onnx model batch size {model_batch_size}, \
                      will use model batch size {model_batch_size}'
            )
            args.batch_size = model_batch_size

        del model

        provider_options = None
        if 'CUDAExecutionProvider' in ort.get_available_providers() and not force_use_cpu:
            providers = (['CUDAExecutionProvider'])
            self.logger.info('Use CUDA device for inference')

        elif 'ROCMExecutionProvider' in ort.get_available_providers() and not force_use_cpu:
            providers = (['ROCMExecutionProvider'])
            self.logger.info('Use ROCM device for inference')

        elif "OpenVINOExecutionProvider" in ort.get_available_providers() and not force_use_cpu:
            providers = (["OpenVINOExecutionProvider"])
            provider_options = [{'device_type': "GPU_FP32"}]
            self.logger.info('Use OpenVINO device for inference')

        else:
            if force_use_cpu:
                self.logger.warning('force_use_cpu ENABLED, will only use cpu for inference!')

            else:
                self.logger.info('Using CPU for inference')
            providers = (['CPUExecutionProvider'])

        ort_infer_sess = ort.InferenceSession(
            model_path,
            providers=providers,
            provider_options=provider_options
        )
        self.logger.info('ONNX model loaded')

        return ort_infer_sess

    def load_csv(
            self,
            tags_csv_path: Path,
    ) -> tuple[list[str], list[str], list[str]]:
        if not os.path.exists(tags_csv_path):
            self.logger.error(f'{str(tags_csv_path)} NOT FOUND!')
            raise FileNotFoundError

        self.logger.info(f'Loading tags from {tags_csv_path}')
        with open(tags_csv_path, 'r', encoding='utf-8') as csv_file:
            csv_content = csv.reader(csv_file)
            rows = [row for row in csv_content]
            header = rows[0]
            rows = rows[1:]

        if not (header[0] == "tag_id" and header[1] == "name" and header[2] == "category"):
            self.logger.error(f'Unexpected csv header: {header}')
            raise ValueError

        rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
        general_tags = [row[1] for row in rows[0:] if row[2] == "0"]
        character_tags = [row[1] for row in rows[0:] if row[2] == "4"]

        return rating_tags, general_tags, character_tags

    def preprocess_tags(
            self,
            rating_tags: list,
            general_tags: list,
            character_tags: list,
    ) -> tuple[list[Any] | list, list[Any] | list, list[Any] | list]:
        if args.character_tag_expand:
            self.logger.info(
                'character_tag_expand Enabled. character tags will be expanded like `character_name, series`.')

            for i, tag in enumerate(character_tags):
                if tag.endswith(")"):
                    tags = tag.split("(")
                    character_tag = "(".join(tags[:-1])

                    if character_tag.endswith("_"):
                        character_tag = character_tag[:-1]
                    series_tag = tags[-1].replace(")", "")

                    character_tags[i] = character_tag + args.caption_separator + series_tag

        if args.remove_underscore:
            self.logger.info('remove_underscore Enabled. `_` will be replace to ` `.')
            rating_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                           rating_tags]
            general_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                            general_tags]
            character_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                              character_tags]

        if args.tag_replacement is not None:
            # escape , and ; in tag_replacement: wd14 tag names may contain , and ;
            escaped_tag_replacements = args.tag_replacement.replace("\\,", "@@@@").replace("\\;", "####")
            tag_replacements = escaped_tag_replacements.split(";")

            for tag_replacement in tag_replacements:
                tags = tag_replacement.split(",")  # source, target

                if not len(tags) == 2:
                    self.logger.error(
                        f'tag replacement must be in the format of `source,target` : {args.tag_replacement}')
                    raise ValueError

                source, target = [tag.replace("@@@@", ",").replace("####", ";") for tag in tags]
                self.logger.info(f'replacing tag: {source} -> {target}')

                if source in general_tags:
                    general_tags[general_tags.index(source)] = target

                elif source in character_tags:
                    character_tags[character_tags.index(source)] = target

                elif source in rating_tags:
                    rating_tags[rating_tags.index(source)] = target

        return rating_tags, general_tags, character_tags

    def run_inference(
            self,
            train_datas_dir: Path,
            ort_infer_sess: InferenceSession,
            rating_tags: list,
            character_tags: list,
            general_tags: list,
    ):
        caption_separator = args.caption_separator
        stripped_caption_separator = caption_separator.strip()
        undesired_tags = args.undesired_tags.split(stripped_caption_separator)
        undesired_tags = set([tag.strip() for tag in undesired_tags if tag.strip() != ""])

        always_first_tags = [tag for tag in args.always_first_tags.split(stripped_caption_separator)
                             if tag.strip() != ""] if args.always_first_tags is not None else None

        # Get image paths
        path_to_find = os.path.join(train_datas_dir, '**') if args.recursive else os.path.join(train_datas_dir, '*')
        image_paths = sorted(set(
            [image for image in glob.glob(path_to_find, recursive=args.recursive)
             if image.lower().endswith(SUPPORT_IMAGE_FORMATS)]),
            key=lambda filename: (os.path.splitext(filename)[0])
        ) if not os.path.isfile(train_datas_dir) else str(train_datas_dir) \
            if str(train_datas_dir).lower().endswith(SUPPORT_IMAGE_FORMATS) else None

        if image_paths is None:
            self.logger.error('Invalid dir or image path!')
            raise FileNotFoundError

        self.logger.info(f'Found {len(image_paths)} image(s).')

        model_shape_size = ort_infer_sess.get_inputs()[0].shape[1]
        input_name = ort_infer_sess.get_inputs()[0].name
        label_name = ort_infer_sess.get_outputs()[0].name

        tag_freq = {}

        # def mcut_threshold(probs: list):
        #     """
        #     Maximum Cut Thresholding (MCut)
        #     Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
        #     for Multi-label Classification. In 11th International Symposium, IDA 2012
        #     (pp. 172-183).
        #     """
        #     probs = numpy.array([x[1] for x in probs])
        #     sorted_probs = probs[probs.argsort()[::-1]]
        #     difs = sorted_probs[:-1] - sorted_probs[1:]
        #     t = difs.argmax()
        #     mcut_threshold = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        #     return mcut_threshold

        def batch_run(path_imgs):
            imgs = numpy.array([im for _, im in path_imgs])

            probs = ort_infer_sess.run([label_name], {input_name: imgs})[0]  # onnx output numpy
            probs = probs[: len(path_imgs)]

            self.logger.debug(
                f'threshold: {args.threshold}') \
                if args.general_threshold is None and args.general_threshold is None else None
            self.logger.debug(
                f'General threshold: {args.general_threshold}') if args.general_threshold is not None else None
            self.logger.debug(
                f'Character threshold: {args.character_threshold}') if args.character_threshold is not None else None

            # Set general_threshold and character_threshold to general_threshold if not they are not set
            args.general_threshold = args.threshold if args.general_threshold is None else args.general_threshold
            args.character_threshold = args.threshold if args.character_threshold is None else args.character_threshold

            for (image_path, _), prob in zip(path_imgs, probs):
                # if args.maximum_cut_threshold:
                #     self.logger.debug('maximum_cut_threshold ENABLED!, all threshold will be overwritten.')
                #     general_prob = prob[len(rating_tags):len(rating_tags)+len(general_tags)]
                #     general_prob = list(zip(general_tags, general_prob.astype(float)))

                #     character_prob = prob[len(rating_tags)+len(general_tags):]
                #     character_prob = list(zip(character_tags, character_prob.astype(float)))

                #     general_threshold = mcut_threshold(general_prob)
                #     self.logger.debug(f'general_threshold changed from '
                #                       f'{args.general_threshold} to {general_threshold}')

                #     character_threshold = mcut_threshold(character_prob)
                #     self.logger.debug(f'character_threshold changed from '
                #                       f'{args.character_threshold} to {character_threshold}')

                combined_tags = []
                rating_tag_text = ""
                character_tag_text = ""
                general_tag_text = ""

                # First 4 labels are ratings, the rest are tags: pick anywhere prediction confidence >= threshold
                for i, p in enumerate(prob[len(rating_tags):]):
                    if i < len(general_tags) and p >= args.general_threshold:
                        tag_name = general_tags[i]

                        if tag_name not in undesired_tags:
                            if args.tags_frequency:
                                tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1

                            general_tag_text += caption_separator + tag_name
                            combined_tags.append(tag_name)

                    elif i >= len(general_tags) and p >= args.character_threshold:
                        tag_name = character_tags[i - len(general_tags)]

                        if tag_name not in undesired_tags:
                            if args.tags_frequency:
                                tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1

                            character_tag_text += caption_separator + tag_name

                            if args.character_tags_first:  # insert to the beginning
                                combined_tags.insert(0, tag_name)

                            else:
                                combined_tags.append(tag_name)

                # First 4 labels are actually ratings: pick one with argmax
                if args.add_rating_tags_to_first or args.add_rating_tags_to_last:
                    ratings_probs = prob[:4]
                    rating_index = ratings_probs.argmax()
                    found_rating = rating_tags[rating_index]

                    if found_rating not in undesired_tags:
                        if args.tags_frequency:
                            tag_freq[found_rating] = tag_freq.get(found_rating, 0) + 1
                        rating_tag_text = found_rating
                        if args.add_rating_tags_to_first:
                            combined_tags.insert(0, found_rating)  # insert to the beginning
                        else:
                            combined_tags.append(found_rating)

                # Always put some tags at the beginning
                if always_first_tags is not None:
                    for tag in always_first_tags:
                        if tag in combined_tags:
                            combined_tags.remove(tag)
                            combined_tags.insert(0, tag)

                if len(general_tag_text) > 0:
                    general_tag_text = general_tag_text[len(caption_separator):]

                if len(character_tag_text) > 0:
                    character_tag_text = character_tag_text[len(caption_separator):]

                if args.custom_caption_save_path is not None:
                    if not os.path.exists(args.custom_caption_save_path):
                        self.logger.error(f'{args.custom_caption_save_path} NOT FOUND!')
                        raise FileNotFoundError

                    self.logger.debug(f'Caption file(s) will be saved in {args.custom_caption_save_path}')

                    if os.path.isfile(train_datas_dir):
                        caption_file = str(os.path.splitext(os.path.basename(image_path))[0])

                    else:
                        caption_file = os.path.splitext(str(image_path).lstrip(str(train_datas_dir)))[0]

                    caption_file = os.path.join(args.custom_caption_save_path, caption_file) + args.caption_extension

                    # Make dir if not exist.
                    os.makedirs(str(caption_file).rstrip(os.path.basename(caption_file)), exist_ok=True)

                else:
                    caption_file = os.path.splitext(image_path)[0] + args.caption_extension

                tag_text = caption_separator.join(combined_tags)

                if args.append_tags:
                    # Check if file exists
                    if os.path.exists(caption_file):
                        with open(caption_file, "rt", encoding="utf-8") as f:
                            # Read file and remove new lines
                            existing_content = f.read().strip("\n")  # Remove newlines

                        # Split the content into tags and store them in a list
                        existing_tags = [tag.strip() for tag in existing_content.split(stripped_caption_separator) if
                                         tag.strip()]

                        # Check and remove repeating tags in tag_text
                        new_tags = [tag for tag in combined_tags if tag not in existing_tags]

                        # Create new tag_text
                        tag_text = caption_separator.join(existing_tags + new_tags)

                if args.not_overwrite and os.path.isfile(caption_file):
                    self.logger.warning(f'Caption file {caption_file} already exist! Skip this caption.')
                    continue

                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(tag_text + "\n")
                    self.logger.debug(f"\tImage path: {image_path}")
                    self.logger.debug(f"\tCaption path: {caption_file}")
                    self.logger.debug(f"\tRating tags: {rating_tag_text}")
                    self.logger.debug(f"\tCharacter tags: {character_tag_text}")
                    self.logger.debug(f"\tGeneral tags: {general_tag_text}")

        # Process images to list for batch run
        pending_imgs = []
        pbar = tqdm(total=len(image_paths), smoothing=0.0)
        for image_path in image_paths:
            try:
                image = Image.open(image_path)
                image = image_process(image, model_shape_size)
                pbar.set_description('Processing: {}'.format(image_path if len(image_path) <= 40 else
                                                             image_path[:15]) + ' ... ' + image_path[-20:])
                pbar.update(1)

            except Exception as e:
                self.logger.error(f"Could not load image path: {image_path}, skip it.\nerror info: {e}")
                continue

            pending_imgs.append((image_path, image))

            if len(pending_imgs) >= args.batch_size:
                pending_imgs = [(str(image_path), image) for image_path, image in
                                pending_imgs]  # Convert image_path to string
                batch_run(pending_imgs)
                pending_imgs.clear()

        if len(pending_imgs) > 0:
            pending_imgs = [(str(image_path), image) for image_path, image in
                            pending_imgs]  # Convert image_path to string
            batch_run(pending_imgs)
            pending_imgs.clear()

        pbar.close()

        if args.tags_frequency:
            sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
            self.logger.info('\tTag frequencies:')
            for tag, freq in sorted_tags:
                self.logger.info(f'\t{tag}: {freq}')

        self.logger.info("Work done!")

    def unload_model(
            self,
            ort_infer_sess: InferenceSession,
            # rating_tags: list,
            # general_tags: list,
            # character_tags: list
    ) -> bool:
        unloaded = False

        if ort_infer_sess is not None:
            del ort_infer_sess
            self.logger.info('ONNX model has been unloaded.')
            # if rating_tags is not None:
            #     del rating_tags
            # if general_tags is not None:
            #     del general_tags
            # if character_tags is not None:
            #     del character_tags
            unloaded = True

        return unloaded


def main(args):
    # Set logger
    workspace_path = os.getcwd()
    data_dir_path = Path(args.data_dir_path)
    log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    caption_failed_list_file = f'Caption_failed_list_{log_time}.txt'

    if os.path.exists(data_dir_path):
        log_name = os.path.basename(data_dir_path)

    else:
        print(f'{data_dir_path} NOT FOUND!!!')
        raise FileNotFoundError

    if args.save_logs:
        log_file = f'Caption_{log_name}_{log_time}.log' if log_name else f'test_{log_time}.log'
        log_file = os.path.join(log_file_path, log_file) \
            if os.path.exists(log_file_path) else os.path.join(os.getcwd(), log_file)
    else:
        log_file = None

    if str(args.log_level).lower() in 'debug, info, warning, error, critical':
        my_logger = Logger(args.log_level, log_file).logger
        my_logger.info(f'Set log level to "{args.log_level}"')

    else:
        my_logger = Logger('INFO', log_file).logger
        my_logger.warning('Invalid log level, set log level to "INFO"!')

    if args.save_logs:
        my_logger.info(f'Log file will be saved as "{log_file}".')

    # Init tagger class
    my_tagger = Tagger(my_logger)

    if args.custom_onnx_path is not None and args.custom_csv_path is not None:
        # Use custom model and csv path
        my_logger.warning('custom_onnx_path and custom_csv_path are enabled')
        if not (os.path.isfile(args.custom_onnx_path) and str(args.custom_onnx_path).endswith('.onnx')):
            my_logger.error(f'{args.custom_onnx_path} is not a onnx file!')
            raise FileNotFoundError

        elif not (os.path.isfile(args.custom_csv_path) and str(args.custom_csv_path).endswith('.csv')):
            my_logger.error(f'{args.custom_csv_path} is not a csv file!')
            raise FileNotFoundError

        model_path, tags_csv_path = args.custom_onnx_path, args.custom_csv_path

    else:
        if args.custom_onnx_path is not None and args.custom_csv_path is None:
            my_logger.warning(f'custom_onnx_path has been set, but custom_csv_path not set. Will ignore these setting!')
        elif args.custom_onnx_path is None and args.custom_csv_path is not None:
            my_logger.warning(f'custom_csv_path has been set, but custom_onnx_path not set. Will ignore these setting!')

        # Download tagger model and csv
        if os.path.exists(Path(args.models_save_path)):
            models_save_path = Path(args.models_save_path)

        else:
            models_save_path = Path(os.path.join(Path(__file__).parent, args.models_save_path))

        # config_file = Path(args.models_config)
        model_path, tags_csv_path = my_tagger.download(
            # config_file = config_file,
            model_name=str(args.model_name),
            model_site=str(args.model_site),
            models_save_path=models_save_path,
            use_sdk_cache=True if args.use_sdk_cache else False,
            download_method=str(args.download_method)
        )

    # Load model
    ort_infer_sess = my_tagger.load_model(
        model_path=model_path,
        force_use_cpu=args.force_use_cpu
    )

    # Load tags from csv
    rating_tags, general_tags, character_tags = my_tagger.load_csv(tags_csv_path)

    # preprocess tags in advance
    rating_tags, general_tags, character_tags = my_tagger.preprocess_tags(rating_tags, general_tags, character_tags)

    # Inference
    my_tagger.run_inference(data_dir_path, ort_infer_sess, rating_tags, character_tags, general_tags)

    # Unload model
    my_tagger.unload_model(ort_infer_sess=ort_infer_sess)


def setup_args() -> argparse.ArgumentParser:
    args = argparse.ArgumentParser()

    args.add_argument(
        'data_dir_path',
        type=str,
        help='path for data dir.'
    )
    args.add_argument(
        '--recursive',
        action='store_true',
        help='Include recursive dirs'
    )
    # args.add_argument(
    #     '--models_config',
    #     type = str,
    #     default = 'config/default.json',
    #     help = 'config json for tagger models, default is "config/default.json"'
    # )
    args.add_argument(
        '--force_use_cpu',
        action='store_true',
        help='force use cpu for inference.'
    )
    args.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='batch size for inference, default is 1.'
    )
    args.add_argument(
        '--model_name',
        type=str,
        default='wd-swinv2-v3',
        help='tagger model name will be used for caption inference, default is "wd-swinv2-v3".'
    )
    args.add_argument(
        '--model_site',
        type=str,
        choices=['huggingface', 'modelscope'],
        default='huggingface',
        help='download model from model site huggingface or modelscope, default is "huggingface".'
    )
    args.add_argument(
        '--models_save_path',
        type=str,
        default="models",
        help='path to save models, default is "models".'
    )
    args.add_argument(
        '--use_sdk_cache',
        action='store_true',
        help='use sdk\'s cache dir to store models. \
            if this option enabled, "--models_save_path" will be ignored.'
    )
    args.add_argument(
        '--download_method',
        type=str,
        choices=["SDK", "URL"],
        default='SDK',
        help='download method via SDK or URL, default is "SDK".'
    )
    args.add_argument(
        '--custom_onnx_path',
        type=str,
        default=None,
        help='Input custom onnx model path, you should use --custom_csv_path together, otherwise this will be ignored'
    )
    args.add_argument(
        '--custom_csv_path',
        type=str,
        default=None,
        help='Input custom tags csv path, you should use --custom_onnx_path together, otherwise this will be ignored'
    )
    args.add_argument(
        '--custom_caption_save_path',
        type=str,
        default=None,
        help='Input custom caption file save path.'
    )
    args.add_argument(
        '--log_level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='set log level, default is "INFO"'
    )
    args.add_argument(
        '--save_logs',
        action='store_true',
        help='save log file.'
    )
    args.add_argument(
        '--caption_extension',
        type=str,
        default='.txt',
        help='extension of caption file, default is ".txt"'
    )
    args.add_argument(
        '--append_tags',
        action='store_true',
        help='Append tags to caption file if existed.'
    )
    args.add_argument(
        '--not_overwrite',
        action='store_true',
        help='not overwrite caption file if exist.'
    )
    args.add_argument(
        '--remove_underscore',
        action='store_true',
        help='replace underscores with spaces in the output tags.',
    )
    args.add_argument(
        "--undesired_tags",
        type=str,
        default='',
        help='comma-separated list of undesired tags to remove from the output.'
    )
    args.add_argument(
        '--tags_frequency',
        action='store_true',
        help='Show frequency of tags for images.'
    )
    args.add_argument(
        '--threshold',
        type=float,
        default=0.35,
        help='threshold of confidence to add a tag, default value is 0.35.'
    )
    args.add_argument(
        '--general_threshold',
        type=float,
        default=None,
        help='threshold of confidence to add a tag from general category, same as --threshold if omitted.'
    )
    args.add_argument(
        '--character_threshold',
        type=float,
        default=None,
        help='threshold of confidence to add a tag for character category, same as --threshold if omitted.'
    )
    # args.add_argument(
    #     '--maximum_cut_threshold',
    #     action = 'store_true',
    #     help = 'Enable Maximum Cut Thresholding, will overwrite every threshold value by its calculate value.'
    # )
    args.add_argument(
        '--add_rating_tags_to_first',
        action='store_true',
        help='Adds rating tags to the first.',
    )
    args.add_argument(
        '--add_rating_tags_to_last',
        action='store_true',
        help='Adds rating tags to the last.',
    )
    args.add_argument(
        '--character_tags_first',
        action='store_true',
        help='Always put character tags before the general tags.',
    )
    args.add_argument(
        '--always_first_tags',
        type=str,
        default=None,
        help='comma-separated list of tags to always put at the beginning, e.g. `1girl,solo`'
    )
    args.add_argument(
        '--caption_separator',
        type=str,
        default=', ',
        help='Separator for captions(include space if needed), default is `, `.'
    )
    args.add_argument(
        '--tag_replacement',
        type=str,
        default=None,
        help='tag replacement in the format of `source1,target1;source2,target2; ...`. '
             'Escape `,` and `;` with `\\`. e.g. `tag1,tag2;tag3,tag4`',
    )
    args.add_argument(
        '--character_tag_expand',
        action='store_true',
        help='expand tag tail parenthesis to another tag for character tags. e.g. '
             '`character_name_(series)` will be expanded to `character_name, series`.',
    )

    return args


if __name__ == "__main__":
    args = setup_args()
    args = args.parse_args()
    main(args)
