# DDA4210-project - SAM(Simpson Artistic Memory)

Group Name: Original Logic

Group Member: Huihan Yang; Jinrui Lin; Rongxiao Qu; Haoming Mo

**Note this repo is still UNDER MAINTAINACE**

## MODEL 

Our models can be found in 🤗[JerryMo/db-simpsons-asim-style](https://huggingface.co/JerryMo/db-simpsons-asim-style) and 🤗[Foxintohumanbeing/simpson-lora](https://huggingface.co/Foxintohumanbeing/simpson-lora). 

`JerryMo/db-simpsons-asim-style` is fine-tuned from SAM and `Foxintohumanbeing/simpson-lora` is fine-tuned from LoRA, whose performance is also ok.

The QR code of our APP is here(APP_QR.png)!Enjoy!👋


**Model Parameter**

The fine-tune parameters is stored in `stored_parameters_for_models`. Here we provide three fine-tune results. 

* The fine-tuned parameters of LoRA is stored in `stored_parameters_for_models\sd-model-lora\pytorch_lora_weights.bin`.

* The fine-tuned parameters of Dreambooth is stored in [GoogleDrive](https://drive.google.com/file/d/1aoCCOsFvzrG27AZ46_kfyx04GGZQkwHF/view?usp=share_link) due to its large size.

* The fine-tuned parameters of **SAM** is stored in [GoogleDrive](https://drive.google.com/file/d/1K4E6b0yqoj95H7Veax8UPorBC1xjaoed/view?usp=share_link) due to its large size.


## Data 

* We preprocess the data (for this part the details will be provided later) and make our own dataset 🤗[JerryMo/image-caption-blip-for-training](https://huggingface.co/datasets/JerryMo/image-caption-blip-for-training). The dataset contains around 2500 pictures with 135MB.

* We also create a [Dataset_App](https://github.com/RickLin616/sd-annotation-app) for give better prompts for pictures. We manually captioned 1000 images and make the second dataset 🤗[JerryMo/Modified-Caption-Train-Set](https://huggingface.co/datasets/JerryMo/Modified-Caption-Train-Set).

* For Dreambooth model, it need caption in great detail but smaller sample size. So we create the third dataset specifically for Dreambooth model 🤗[JerryMo/db-simpsons-dataset](https://huggingface.co/datasets/JerryMo/db-simpsons-dataset). Notice that we use 'Asim' as keyword in the caption.

* For less fine-tuning time, you may also need a dataset with smaller size. Here are two ways you could do:
    
    1. Tune the parameter max_train_samples.

    2. Use 🤗[Skiracer/simpsons_blip_captions](https://huggingface.co/datasets/skiracer/simpsons_blip_captions), which is a relatively small dataset.


## Trainning 

* For preprocessing and requirement of packages, you may need to refer to 🤗[huggingface/diffusers](https://github.com/huggingface/diffusers). 

```
git clone https://github.com/foxintohumanbeing/DDA4210_Group_project.git
```

1. Our model is fine-tuned on 🤗 CompVis/stable-diffusion-v1-4.

2. Files are stored in `fine_tuning_files`

*  `fine_tuning_files/train_dreambooth_lora_unfreezed.py`: code of SAM model.

* `fine_tuning_files/train_dreambooth_lora.py`: code of model utilizing LoRA in DreamBooth.

*  `fine_tuning_files/train_dreambooth.py`: code fine-tuning simply use DreamBooth method.

*  `fine_tuning_files/train_text_to_image_lora.py`: code fine-tuning simply use LoRA method.

*  `fine_tuning_files/train_text_to_image.py`: code fine-tuning without any technique.

3. `configuration_file/config_train.json` stores the parameters you need to change. 


**Testing Command**
```
cd DDA4210_Group_project
python fine_tuning_files/train_dreambooth_lora_unfreezed.py --config_path="configuration_file\config_train.json"
```

## Inferencing

1. Files are stored in `fine_tuning_files`

*  `fine_tuning_files/inference_dreambooth_lora_unet.py`: code of SAM model.

*  `fine_tuning_files/inference_dreambooth_lora.py`: code of model utilizing LoRA in DreamBooth.

*  `fine_tuning_files/inference_dreambooth.py`: code fine-tuning simply use DreamBooth method.

*  `fine_tuning_files/inference_lora.py`: code fine-tuning simply use LoRA method.

*  `fine_tuning_files/inference_simple.py`: code fine-tuning without any technique.

2. `configuration_file/config_test.json` stores the parameters you need to change. 

PLEASE make sure that the parameter `output_dir` and `pretrained_model_name_or_pat`h is the SAME as the parameter `output_dir`` and pretrained_model_name_or_path` in `config_train.json`. 

**Training Command**
```
python fine_tuning_files/inference_dreambooth_lora_unet.py --config_path="configuration_file\config_test.json"
```

(other is still under process)

## How to measure?🤔

### Frechet Inception Distance (FID)

Instructions can be found in [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid).

### Language Drifting Measurement (LDM)



For any questions, please CONTACT Huihan Yang ASAP!
