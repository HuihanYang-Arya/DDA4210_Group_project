# DDA4210-project - SAM(Simpson Artistic Memory)

Group Name: Original Logic

Group Member: Huihan Yang; Jinrui Lin; Rongxiao Qu; Haoming Mo

**Note this repo is still UNDER MAINTAINACE**

## MODEL 

Our model can be found in 🤗 https://huggingface.co/JerryMo/db-simpsons-asim-style and 🤗 https://huggingface.co/Foxintohumanbeing/simpson-lora. 

The QR code of our APP is here!

[QRcode for our application](APP_QR.png)

Enjoy!👋

* The fine-tune parameters is stored in `stored_parameters_for_models`. Here we provide three fine-tune results. 

    * The fine-tuned parameters of LoRA is stored in `stored_parameters_for_models\sd-model-lora\pytorch_lora_weights.bin`.

    * The fine-tuned parameters of Dreambooth is stored in https://drive.google.com/file/d/1aoCCOsFvzrG27AZ46_kfyx04GGZQkwHF/view?usp=share_link due to its large size.

    * The fine-tuned parameters of **SAM** is stored in https://drive.google.com/file/d/1K4E6b0yqoj95H7Veax8UPorBC1xjaoed/view?usp=share_link due to its large size.


## Data 

* We preprocess the data (for this part the details will be provided later) and make our own dataset https://huggingface.co/datasets/JerryMo/image-caption-blip-for-training. The dataset contains around 2500 pictures with 135MB.

* We also create a **dataset app** for give better prompts for pictures, please check https://github.com/RickLin616/sd-annotation-app

* For less fine-tuning time, you may also need a dataset with smaller size. Here are two ways you could do:
    
    1. Tune the parameter max_train_samples.

    2. Use https://huggingface.co/datasets/skiracer/simpsons_blip_captions, which is a relatively small dataset.


## Trainning (UNDER MAINTAINANCE)

PLEASE NOTE THAT FOR PREPROCESSING AND REQUIREMENTS OF PACKAGES, you may need to refer to https://github.com/huggingface/diffusers and https://github.com/huggingface/diffusers/tree/main/examples/text_to_image. 

Our model is fine-tuned on 🤗 CompVis/stable-diffusion-v1-4.

Files are stored in `fine_tuning_files`

*  `train_dreambooth_lora_unfreezed.py` contains the code of our fine-tuning model.

*  `train_dreambooth.py` contains the code fine-tuning simply use dreambooth method.

*  `train_text_to_image_lora.py` contains the code fine-tuning simply use lora method.

*  `train_text_to_image.py` contains the code fine-tuning without any technique.


To run the training, `python train_text_to_image.py` 

(please note that dist_training is still under-maintained)

## Inferencing

1. `configuration_file/config_test.json` stores the parameter you need to change. 

PLEASE make sure that the parameter output_dir and pretrained_model_name_or_path is the SAME as the parameter output_dir and pretrained_model_name_or_path in `config.json`. 

To change the prompts, please change the content in prompts in the format of list. parameter num_images refers to the number of pictures generated of each prompt. Please be aware that to generate a simpson's style picture, word "The Simpson" is required in the last of sentence. 

2. to run the testing, `python inference_lora.py`

(other is still under process)

## measurement of fine-tune algorithm

1. Frechet Inception Distance (FID)

    Instructions can be found in https://github.com/mseitzer/pytorch-fid.

2. Crossed Eyes Ratio

    This one is calculated with hands.

3. LDM
    
    We use two pretrained model: images-Inceptionv3 and text-CLIP to get the embedding vectors and calculate their cosine similarity.



For any questions, please CONTACT Huihan Yang ASAP!
