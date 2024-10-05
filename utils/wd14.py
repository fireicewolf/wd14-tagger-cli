import argparse
import csv
import os
import time
from pathlib import Path

import numpy
from PIL import Image
from tqdm import tqdm

from utils.image import image_process, image_process_gbr, get_image_paths
from utils.logger import Logger

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
    def __init__(
            self,
            logger: Logger,
            args: argparse.Namespace,
            model_path: Path,
            tags_csv_path: Path
    ):
        self.logger = logger
        self.args = args

        self.ort_infer_sess = None
        self.model_name = None
        self.model_path = model_path
        self.tags_csv_path = tags_csv_path

        self.rating_tags = None
        self.character_tags = None
        self.general_tags = None

    def load_model(self):
        args = self.args
        if not os.path.exists(self.model_path):
            self.logger.error(f'{str(self.model_path)} NOT FOUND!')
            raise FileNotFoundError

        import onnx
        import onnxruntime as ort

        self.logger.info(f'Loading model from {str(self.model_path)}')
        model = onnx.load(self.model_path)

        # get model batch size info
        try:
            model_batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
        except:
            model_batch_size = model.graph.input[0].type.tensor_type.shape.dim[0].dim_param

        if args.batch_size != model_batch_size and not isinstance(model_batch_size, str) and model_batch_size > 0:
            # some re-batch model may use 'N' as dynamic axes
            self.logger.warning(
                f'Batch size {args.batch_size} don\'t match onnx model batch size {model_batch_size}, \
                      will use model batch size {model_batch_size}'
            )
            args.batch_size = model_batch_size

        del model

        provider_options = None
        if 'CUDAExecutionProvider' in ort.get_available_providers() and not args.force_use_cpu:
            providers = (['CUDAExecutionProvider'])
            self.logger.info('Use CUDA device for inference')

        elif 'ROCMExecutionProvider' in ort.get_available_providers() and not args.force_use_cpu:
            providers = (['ROCMExecutionProvider'])
            self.logger.info('Use ROCM device for inference')

        elif "OpenVINOExecutionProvider" in ort.get_available_providers() and not args.force_use_cpu:
            providers = (["OpenVINOExecutionProvider"])
            provider_options = [{'device_type': "GPU_FP32"}]
            self.logger.info('Use OpenVINO device for inference')

        else:
            if args.force_use_cpu:
                self.logger.warning('force_use_cpu ENABLED, will only use cpu for inference!')

            else:
                self.logger.info('Using CPU for inference')
                args.force_use_cpu = True
            providers = (['CPUExecutionProvider'])

        self.logger.info(f'Loading {args.model_name} with {"CPU" if args.force_use_cpu else "GPU"}...')
        start_time = time.monotonic()

        self.model_name = args.model_name

        self.ort_infer_sess = ort.InferenceSession(
            self.model_path,
            providers=providers,
            provider_options=provider_options
        )

        self.logger.info(f'{args.model_name} Loaded in {time.monotonic() - start_time:.1f}s.')

    def load_csv(self):
        tags_csv_path = self.tags_csv_path
        if not os.path.exists(tags_csv_path):
            self.logger.error(f'{str(tags_csv_path)} NOT FOUND!')
            raise FileNotFoundError

        self.logger.info(f'Loading tags from {tags_csv_path}')
        with open(tags_csv_path, 'r', encoding='utf-8') as csv_file:
            csv_content = csv.reader(csv_file)
            rows = [row for row in csv_content]
            header = rows[0]
            rows = rows[1:]

        if not (header[0] in ("tag_id", "id") and header[1] == "name" and header[2] == "category"):
            self.logger.error(f'Unexpected csv header: {header}')
            raise ValueError

        model_name = str(self.model_name)
        if model_name.lower().startswith("wd"):
            rating_tags = [row[1] for row in rows[0:] if row[2] == "9"]
            character_tags = [row[1] for row in rows[0:] if row[2] == "4"]
            general_tags = [row[1] for row in rows[0:] if row[2] == "0"]

        else:
            self.logger.warning(f"{model_name} doesn't support rating tags and character tags.")
            rating_tags = None
            character_tags = None
            general_tags = [row[1] for row in rows[0:]]

        self.rating_tags = rating_tags
        self.character_tags = character_tags
        self.general_tags = general_tags

    def preprocess_tags(self):
        args = self.args
        model_name = str(self.model_name)

        rating_tags = self.rating_tags
        character_tags = self.character_tags
        general_tags = self.general_tags

        if args.character_tag_expand:
            if model_name.lower().startswith("wd"):
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
            else:
                self.logger.warning(f"{model_name} doesn't support and character tags.")

        if args.remove_underscore:
            self.logger.info('remove_underscore Enabled. `_` will be replace to ` `.')
            if model_name.lower().startswith("wd"):
                rating_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                               rating_tags]

                character_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                                  character_tags]

            general_tags = [tag.replace("_", " ") if len(tag) > 3 and tag not in kaomojis else tag for tag in
                            general_tags]

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

                elif source in character_tags and model_name.lower().startswith("wd"):
                    character_tags[character_tags.index(source)] = target

                elif source in rating_tags and model_name.lower().startswith("wd"):
                    rating_tags[rating_tags.index(source)] = target

        self.rating_tags = rating_tags
        self.character_tags = character_tags
        self.general_tags = general_tags

    def run_inference(self):
        args = self.args
        train_datas_dir = args.data_dir_path
        ort_infer_sess = self.ort_infer_sess

        rating_tags = self.rating_tags
        character_tags = self.character_tags
        general_tags = self.general_tags

        caption_separator = args.caption_separator
        stripped_caption_separator = caption_separator.strip()
        undesired_tags = args.undesired_tags.split(stripped_caption_separator)
        undesired_tags = set([tag.strip() for tag in undesired_tags if tag.strip() != ""])

        always_first_tags = [tag for tag in args.always_first_tags.split(stripped_caption_separator)
                             if tag.strip() != ""] if args.always_first_tags is not None else None

        # Get image paths
        image_paths = get_image_paths(logger=self.logger, path=Path(self.args.data_dir_path),
                                      recursive=self.args.recursive)

        model_shape_size = ort_infer_sess.get_inputs()[0].shape[1]
        input_name = ort_infer_sess.get_inputs()[0].name
        label_name = ort_infer_sess.get_outputs()[0].name

        self.logger.debug(f'"{self.model_name}" target shape is {model_shape_size}')

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
            model_name = str(args.model_name)

            if not model_name.lower().startswith("wd"):
                self.logger.warning(f'"{model_name}" don\'t support general_threshold and character_threshold, '
                                    f'will set them to threshold value')
                args.general_threshold = None
                args.character_threshold = None

            self.logger.debug(
                f'threshold: {args.threshold}') \
                if args.general_threshold is None and args.character_threshold is None else None
            self.logger.debug(
                f'General threshold: {args.general_threshold}') if args.general_threshold is not None else None
            self.logger.debug(
                f'Character threshold: {args.character_threshold}') if args.character_threshold is not None else None

            # Set general_threshold and character_threshold to general_threshold if not they are not set
            args.general_threshold = args.threshold if args.general_threshold is None else args.general_threshold
            args.character_threshold = args.threshold \
                if args.character_threshold is None and model_name.lower().startswith("wd") \
                else args.character_threshold

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
                for i, p in enumerate(prob[len(rating_tags):] if model_name.lower().startswith("wd") else prob):
                    if i < len(general_tags) and p >= args.general_threshold:
                        tag_name = general_tags[i]

                        if tag_name not in undesired_tags:
                            if args.tags_frequency:
                                tag_freq[tag_name] = tag_freq.get(tag_name, 0) + 1

                            general_tag_text += caption_separator + tag_name
                            combined_tags.append(tag_name)

                    elif (args.character_threshold is not None
                          and i >= len(general_tags) and p >= args.character_threshold):
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
                    if model_name.lower().startswith("wd"):
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
                    else:
                        self.logger.warning(f"{model_name} doesn't support rating tags.")

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
                        caption_file = os.path.splitext(str(image_path)[len(str(train_datas_dir)):])[0]

                    caption_file = caption_file[1:] if caption_file[0] == '/' else caption_file
                    caption_file = os.path.join(args.custom_caption_save_path, caption_file)
                    # Make dir if not exist.
                    os.makedirs(Path(str(caption_file)[:-len(os.path.basename(caption_file))]), exist_ok=True)
                    caption_file = Path(str(caption_file) + args.caption_extension)

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
                    if model_name.lower().startswith("wd"):
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
                image = image_process_gbr(image)
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

    def unload_model(
            self
    ) -> bool:
        unloaded = False
        if self.ort_infer_sess is not None:
            self.logger.info(f'Unloading model {self.model_name}...')
            start = time.monotonic()
            del self.ort_infer_sess
            if self.rating_tags is not None:
                del self.rating_tags
            if self.character_tags is not None:
                del self.character_tags
            if self.general_tags is not None:
                del self.general_tags
            self.logger.info(f'{self.model_name} unloaded in {time.monotonic() - start:.1f}s.')
            del self.model_name
            del self.model_path
            del self.tags_csv_path

            unloaded = True

        return unloaded
