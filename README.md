# DDA4210-project - SAM(Simpson Artistic Memory)

Group Name: Original Logic

Group Member: Huihan Yang; Jinrui Lin; Rongxiao Qu; Haoming Mo

**Note this repo is still UNDER MAINTAINACE**

Link for our basic working pipeline: https://shimo.im/file-invite/Xyip8inwgJqyJqzQRye5NvMZKbpA6/ urbbrgroun 邀请您协作「Diffusion-simpons-4210-project」，点击链接开始编辑 (please contact me if the link is invalid)

## MODEL 

Our model can be found in 🤗 https://huggingface.co/JerryMo/db-simpsons-asim-style and 🤗 https://huggingface.co/Foxintohumanbeing/simpson-lora. The QR code of our APP is in `APP_QR.png`. Enjoy!👋

## Data

* We preprocess the data (for this part the details will be provided later) and make our own dataset https://huggingface.co/datasets/JerryMo/image-caption-blip-for-training. The dataset contains around 2500 pictures with 135MB.

* For more details about dataset app, please check https://github.com/RickLin616/sd-annotation-app

* For convenient training, you may also need a dataset with smaller size. Here are two ways you could do:
    
    1. tune the parameter max_train_samples in `config.json`.

    2. use https://huggingface.co/datasets/skiracer/simpsons_blip_captions, which is a relatively small dataset.


## Trainning

PLEASE NOTE THAT FOR PREPROCESSING AND REQUIREMENT OF PIPELINE, you may need to refer to https://github.com/huggingface/diffusers and https://github.com/huggingface/diffusers/tree/main/examples/text_to_image. 

1. `configuration_file/config.json` stores the parameter you need to change.

For example, if you want to change the pre-trained model or the dataset, just change the content in `config.json`

2. to run the training please type in the following command in terminal
`python train_text_to_image.py` or
`python train_text_to_image_lora.py`

(please note that `train_text_to_image_flax.py` is still under-maintained)

## Inferencing

1. `configuration_file/config_test.json` stores the parameter you need to change. 

PLEASE make sure that the parameter output_dir and pretrained_model_name_or_path is the SAME as the parameter output_dir and pretrained_model_name_or_path in `config.json`. 

To change the prompts, please change the content in prompts in the format of list. parameter num_images refers to the number of pictures generated of each prompt. Please be aware that to generate a simpson's style picture, word "The Simpson" is required in the last of sentence. 

2. to run the testing please type in the following command in terminal
`python inference_lora.py`

(other is still under process)

3. Some fine-tuned models has been provided.

    **file `sd-model-finetuned-114514`**
    * `pytorch_lora_weights.bin` stores the fine-tune results.
    
    * Corresponding configurations can be found in `configuration_file/config_train_114514.json`. To directly see the result, you could change the outdir and pretrained_model_name_or_path parameters in `configuration_file/config_test.json` according to the ones in `configuration_file/config_train_114514.json`. You could also change the prompts to whatever you like. 

## measurement of fine-tune algorithm

1. Frechet Inception Distance (FID)

    * Note that to measure the differences between the original pics and generated pics, the original pics need to be resized into 512*512. To do this, you need to format the picturse into two folders `pics/generated_images` and `pics/real_images`. Then you could directly run `python utils/resized_original_data.py` to get the `pics/resized_original_pic` folder.

    * To run this, please first make sure you install the packages following https://github.com/mseitzer/pytorch-fid. 

    Now you could run `python -m pytorch_fid generated_images resized_original_pic --device cuda:0 --dims 768`. The meaning of parameters and source code can be found in https://github.com/mseitzer/pytorch-fid.

2. Likelihood 

    * link: https://arxiv.org/pdf/2011.13456.pdf


## Result Record

* sd-model-finetuned-114514-large: 2000pics/80epoch/non-modified data/lora

* sd-model-finetuned-lora-regu: lora with regularization technique added

* model-simple-finetuned: 2000pics/80epoch/non-modified data/simple method


For any questions, please CONTACT Huihan Yang ASAP!
