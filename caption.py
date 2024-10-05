import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from utils.download import download_models
from utils.logger import Logger
from utils.wd14 import Tagger


def main(args):
    # Set logger
    workspace_path = os.getcwd()
    data_dir_path = Path(args.data_dir_path)
    log_file_path = data_dir_path.parent if os.path.exists(data_dir_path.parent) else workspace_path

    if args.custom_caption_save_path:
        log_file_path = Path(args.custom_caption_save_path)

    log_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    # caption_failed_list_file = f'Caption_failed_list_{log_time}.txt'

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

    # Check custom models path
    config_file = os.path.join(Path(__file__).parent, 'configs', 'default.json') \
        if args.config == "default.json" else Path(args.config)

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
        model_path, tags_csv_path = download_models(
            logger=my_logger,
            args=args,
            config_file=config_file,
            models_save_path=models_save_path
        )

    # Init tagger class
    my_tagger = Tagger(logger=my_logger, args=args, model_path=model_path, tags_csv_path=tags_csv_path)
    # Load model
    my_tagger.load_model()
    # Load tags from csv
    my_tagger.load_csv()
    # preprocess tags in advance
    my_tagger.preprocess_tags()
    # Inference
    start_inference_time = time.monotonic()
    my_tagger.run_inference()
    total_inference_time = time.monotonic() - start_inference_time
    days = total_inference_time // (24 * 3600)
    total_inference_time %= (24 * 3600)
    hours = total_inference_time // 3600
    total_inference_time %= 3600
    minutes = total_inference_time // 60
    seconds = total_inference_time % 60
    days = f"{days} Day(s) " if days > 0 else ""
    hours = f"{hours} Hour(s) " if hours > 0 or (days and hours == 0) else ""
    minutes = f"{minutes} Min(s) " if minutes > 0 or (hours and minutes == 0) else ""
    seconds = f"{seconds:.1f} Sec(s)"
    my_logger.info(f"All work done with in {days}{hours}{minutes}{seconds}.")
    # Unload model
    my_tagger.unload_model()


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
    args.add_argument(
        '--config',
        type=str,
        default='default.json',
        help='configs json for tagger models, default is "default.json"'
    )
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
        default='wd-eva02-large-tagger-v3',
        help='tagger model name will be used for caption inference, default is "wd-eva02-large-tagger-v3".'
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
        '--force_download',
        action='store_true',
        help='force download even file exists.'
    )
    args.add_argument(
        '--skip_download',
        action='store_true',
        help='skip download if exists.'
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
