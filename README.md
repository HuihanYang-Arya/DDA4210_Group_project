# DDA4210-project

Group Member: Huihan Yang; Jinrui Lin; Rongxiao Qu; Haoming Mo
Link for our basic working pipeline: https://shimo.im/file-invite/Xyip8inwgJqyJqzQRye5NvMZKbpA6/ urbbrgroun 邀请您协作「Diffusion-simpons-4210-project」，点击链接开始编辑 (please contact me if the link is invalid)

## Data

We preprocess the data (for this part the details will be provided later) and make our own dataset https://huggingface.co/datasets/JerryMo/image-caption-blip-for-training. The dataset contains around 2500 pictures with 135MB.

For convenient training, you may also need a dataset with smaller size. Here are two ways you could do:
    
    1. tune the parameter max_train_samples in `config.json`.

    2. use https://huggingface.co/datasets/skiracer/simpsons_blip_captions, which is a relatively small dataset.


## Trainning

PLEASE NOTE THAT FOR PREPROCESSING AND REQUIREMENT OF PIPELINE, you may need to refer to https://github.com/huggingface/diffusers and https://github.com/huggingface/diffusers/tree/main/examples/text_to_image. 

1. `config.json` stores the parameter you need to change.
For example, if you want to change the pre-trained model or the dataset, just change the content in `config.json`

2. to run the training please type in the following command in terminal
`python train_text_to_image.py` or
`python train_text_to_image_lora.py`

(please note that `train_text_to_image_flax.py` is still under-maintained)

## Inferencing
1. `config_test.json` stores the parameter you need to change. 
PLEASE make sure that the parameter output_dir and pretrained_model_name_or_path is the SAME as the parameter output_dir and pretrained_model_name_or_path in `config.json`. 
To change the prompts, please change the content in prompts in the format of list. parameter num_images refers to the number of pictures generated of each prompt.

2. to run the testing please type in the following command in terminal
`python inference.py`


For any questions, please CONTACT Huihan Yang ASAP!