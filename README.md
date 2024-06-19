# wd14-tagger-cli
A Python base cli tool for tagging images with wd14 models.

## Introduce

I make this repo because I want to caption some images cross-platform (On My old MBP, my game win pc or docker base linux cloud-server(like Google colab))

But I don't want to install a huge webui just for this little work. And some cloud-service are unfriendly to gradio base ui.

So this repo born.


## Model source

All models are from [SmilingWolf](https://huggingface.co/SmilingWolf)(👏👏)

Huggingface are original sources, modelscope are pure forks from Huggingface(Because HuggingFace was blocked in Some place).

|            Model             |                                HuggingFace Link                                |                                     ModelScope Link                                     |
|:----------------------------:|:------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|
|     wd-swinv2-tagger-v3      |     [HuggingFace](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3)      |     [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-swinv2-tagger-v3)      |
|       wd-vit-tagger-v3       |       [HuggingFace](https://huggingface.co/SmilingWolf/wd-vit-tagger-v3)       |       [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-vit-tagger-v3)       |
|    wd-convnext-tagger-v3     |    [HuggingFace](https://huggingface.co/SmilingWolf/wd-convnext-tagger-v3)     |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-convnext-tagger-v3)     |
|    wd-v1-4-moat-tagger-v2    |    [HuggingFace](https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2)    |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-moat-tagger-v2)    |
|   wd-v1-4-swinv2-tagger-v2   |   [HuggingFace](https://huggingface.co/SmilingWolf/wd-v1-4-swinv2-tagger-v2)   |   [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-swinv2-tagger-v2)   |
| wd-v1-4-convnextv2-tagger-v2 | [HuggingFace](https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2) | [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-convnextv2-tagger-v2) |
|    wd-v1-4-vit-tagger-v2     |    [HuggingFace](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2)     |    [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-vit-tagger-v2)     |
|  wd-v1-4-convnext-tagger-v2  |  [HuggingFace](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2)  |  [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-convnext-tagger-v2)  |
|      wd-v1-4-vit-tagger      |      [HuggingFace](https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger)      |      [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-vit-tagger)      |
|   wd-v1-4-convnext-tagger    |   [HuggingFace](https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger)    |   [ModelScope](https://www.modelscope.cn/models/fireicewolf/wd-v1-4-convnext-tagger)    |

## TO-DO

make a simple ui by Jupyter widget(When my lazy cancer cured😊)

## Installation
Python 3.10-3.12 works fine. 

Open a shell terminal and follow below steps:
```shell
# Clone this repo
git clone https://github.com/fireicewolf/wd14-tagger-cli.git
cd wd14-tagger-cli

# create a Python venv
python -m venv .venv
.\venv\Scripts\activate

# Install dependencies
# Base dependencies, models for inference will download via python request libs.
pip install -U -r requirements.txt

# If you want to download or cache model via huggingface hub, install this.
pip install -U -r huggingface-requirements.txt

# If you want to download or cache model via modelscope hub, install this.
pip install -U -r modelscope-requirements.txt

# If you want to use cuda devices for inference, install one of these two depend on your CUDA version.
# For CUDA 11.8
pip install -U -r cuda118-requirements.txt
# For CUDA 12.x
pip install -U -r cuda12x-requirements.txt
```

### Take a notice
I have added CUDA, ROCm and OpenVino providers in inference code, but I didn't test if all of them are work(ROCm and OpenVINO).

In code the priority of device for inference is CUDA -> ROCm ->OpenVINO ->CPU.

You may need to install extra sdk or pip package for ROCm or OpenVINO to work,
Please follow this [doc](https://onnxruntime.ai/docs/execution-providers/#summary-of-supported-execution-providers) on [onnxruntime.ai](https://onnxruntime.ai/docs/execution-providers/#summary-of-supported-execution-providers) website.

## Simple usage
__Make sure your python venv has been activated first!__
```shell
python caption.py your_datasets_path
```
To run with more options, You can find help by run with this or see at [Options](#options)
```shell
python caption.py -h
```

##  <span id="options">Options</span>
<details>
    <summary>Advance options</summary>

`--recursive`

Will include all support images format in your input datasets path and its subpath.

`--force_use_cpu`

Force use cpu for inference.

`--batch_size N`

Batch size for inference, default is 1.

`--model_name MODEL_NAME`

Onnx model name used for inference, default is wd-swinv2-v3(For more model, please check config/default.json)

`--model_site MODEL_SITE`

Model site where onnx model download from(huggingface or modelscope), default is huggingface.

`--models_save_path MODEL_SAVE_PATH`

Path for models to save, default is models(under project folder).

`--download_method `

Download models via sdk or url, default is sdk.

If huggingface hub or modelscope sdk not installed or download failed, will auto retry with url download.

`--use_sdk_cache`

Use huggingface or modelscope sdk cache to store models, this option need huggingface_hub or modelscope sdk installed.

If this enabled, `--models_save_path` will be ignored.

`--custom_onnx_path CUSTOM_ONNX_PATH`
`--custom_csv_path CUSTOM_CSV_PATH`

This two args need to be used together. You can use your exist model.

`--custom_caption_save_path CUSTOM_CAPTION_SAVE_PATH`

Save caption files to a custom path but not with images(But keep their directory structure)

`--log_level LOG_LEVEL`

Log level for terminal console and log file, default is `INFO`(`DEBUG`,`INFO`,`WARNING`,`ERROR`,`CRITICAL`)

`--save_logs`

Save logs to a file, log will be saved at same level with `data_dir_path`

`--caption_extension CAPTION_EXTENSION`

Caption file extension, default is `.txt`

`--append_tags APPEND_TAGS`

Append tags to caption file if existed.

`--not_overwrite`

Do not overwrite caption file if it existed.

`--remove_underscore`

Remove "_" symbol in tags(not include kmoji like o_o).

`--undesired_tags UNDESIRED_TAGS`

Tags you don't want appeared in captions, seperate them with comma like `"black,yellow"`

`--tags_frequency`

Enable this will make a statistics of the tags occurred frequency.

`--threshold THRESHOLD`

Threshold of confidence to add a tag to caption, default value is 0.35

`--general_threshold GENERAL_THRESHOLD`

Threshold of confidence to add a tag from general category, if not defined, will use `--threshold` as it.

`--character_threshold CHARACTER_THRESHOLD`

Threshold of confidence to add a tag from character category, if not defined, will use `--threshold` as it.

`--add_rating_tags_to_first`

Add rating tags at the beginning of caption.

`--add_rating_tags_to_last`

Add rating tags at the end of caption.

`--character_tags_first`

Make character_tags to the beginning of caption.

`--always_first_tags ALWAYS_FIRST_TAGS`

Tags(separate with comma like "1boy,solo") you want to put in the beginning of caption.

`--caption_separator CAPTION_SEPARATOR`

Separator for captions(include space if needed), default is `", "`.

`--tag_replacement TAG_REPLACEMENT`

tag replacement in the format of `"source1,target1;source2,target2; ..."`. 

Escape `,` and `;` with `\`. e.g. `"tag1,tag2;tag3,tag4"`

`--character_tag_expand`

Expand tag tail parenthesis to another tag for character tags.
 
e.g. `character_name_(series)` will be expanded to `character_name, series`.

</details>

## Credits
Most tags process code from [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts/blob/main/finetune/tag_images_by_wd14_tagger.py)

Some image proces code from [toriato/stable-diffusion-webui-wd14-tagger](https://github.com/Akegarasu/sd-webui-wd14-tagger/blob/master/tagger/dbimutils.py) 
and [SmilingWolf/wd-tagger](https://huggingface.co/spaces/SmilingWolf/wd-tagger/blob/main/app.py)

Without their works(👏👏), this repo won't exist.