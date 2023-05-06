# DDA4210-project - SAM(Simpson Artistic Memory)

Group Name: Original Logic

Group Member: Huihan Yang; Jinrui Lin; Rongxiao Qu; Haoming Mo

**NO BUSINESS USAGE**

## MODEL 

Our models can be found in 🤗[JerryMo/db-simpsons-asim-style](https://huggingface.co/JerryMo/db-simpsons-asim-style) and 🤗[Foxintohumanbeing/simpson-lora](https://huggingface.co/Foxintohumanbeing/simpson-lora). 

`JerryMo/db-simpsons-asim-style` is fine-tuned from SAM and `Foxintohumanbeing/simpson-lora` is fine-tuned from LoRA, whose performance is also ok.

The QR code of our APP is [here](APP_QR.png)!Enjoy!👋

Sample images can be found in `Sample Generate Imags`.

**Model Checkpoint**

The fine-tune parameters are stored in `stored_parameters_for_models`. Here we provide three fine-tune results. 

* The fine-tuned parameters of LoRA is stored in `stored_parameters_for_models\sd-model-lora\pytorch_lora_weights.bin`.

* The fine-tuned parameters of Dreambooth is stored in [GoogleDrive](https://drive.google.com/file/d/1aoCCOsFvzrG27AZ46_kfyx04GGZQkwHF/view?usp=share_link) due to its large size.

* The fine-tuned parameters of **SAM** is stored in [GoogleDrive](https://drive.google.com/file/d/1K4E6b0yqoj95H7Veax8UPorBC1xjaoed/view?usp=share_link) due to its large size.

**Note**: The checkpoint may not able to be directly utilized in the inference stage using code. You may need to check model's structure on huggingface(as given) for further utilization.


## Data 

* We preprocess the data (for this part the details will be provided later) and make our own dataset 🤗[JerryMo/image-caption-blip-for-training](https://huggingface.co/datasets/JerryMo/image-caption-blip-for-training). The dataset contains around 2500 pictures with 135MB.

* We also create a [Dataset_App](https://github.com/RickLin616/sd-annotation-app) for give better prompts for pictures. We manually captioned 1000 images and make the second dataset 🤗[JerryMo/Modified-Caption-Train-Set](https://huggingface.co/datasets/JerryMo/Modified-Caption-Train-Set).

* For Dreambooth model, it need caption in great detail but smaller sample size. So we create the third dataset specifically for Dreambooth model 🤗[JerryMo/db-simpsons-dataset](https://huggingface.co/datasets/JerryMo/db-simpsons-dataset). Notice that we use 'Asim' as keyword in the caption.

* For less fine-tuning time, you may also need a dataset with smaller size. Here are two ways you could do:
    
    1. Tune the parameter max_train_samples.

    2. Use 🤗[Skiracer/simpsons_blip_captions](https://huggingface.co/datasets/skiracer/simpsons_blip_captions), which is a relatively small dataset.


## Running command

**Note:** Before start, we strongly recommend you to validate our result on hugging face instead of direct coding. `Diffuers` updates frequently so codes need to be kepted updating. Moreover, to use the ckpt file we provided efficiently, we already built API on 🤗Hugging Face. If you want to run inference on your own, you may need to form the folder of ckpt file on your own following the instructions from hugging face documentation.   

Our model is fine-tuned on 🤗[CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4).

**Requirement**

For preprocessing and requirement of packages, you need to refer to the installzation from 🤗[huggingface/diffusers](https://github.com/huggingface/diffusers). For any problem occurs, please first check whether the **versions of huggingface, diffusers, torch and CUDA match**.


```
git clone https://github.com/foxintohumanbeing/DDA4210_Group_project.git
cd DDA4210_Group_project
```

**Training Command**
```
python fine_tuning_files/train/train_dreambooth_lora_unfreezed.py
```

**Testing Command**
```
python fine_tuning_files/inference/inference_dreambooth_lora_unet.py 
```
PLEASE make sure that the parameter `output_dir` and `pretrained_model_name_or_pat` is the SAME as the parameter `output_dir` and `pretrained_model_name_or_path` in `config_train.json`. 

**Measurement Command**

* Frechet Inception Distance (FID)

Instructions can be found in [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid).

* Language Drifting Measurement (LDM)

We use the 🤗[openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14). Realized code can be found in `utils/LDM.py`.


## File Explaination

### Fine-tuning Files
#### Train 

Training codes.

*  `train_dreambooth_lora_unfreezed.py`: code of SAM model.

* `train_dreambooth_lora.py`: code of model utilizing LoRA in DreamBooth.

*  `train_dreambooth.py`: code fine-tuning simply use DreamBooth method.

*  `train_text_to_image_lora.py`: code fine-tuning simply use LoRA method.

*  `train_text_to_image.py`: code fine-tuning without any technique.

#### Inference

1. Files are stored in `fine_tuning_files`

*  `inference_dreambooth_lora_unet.py`: code of SAM model.

*  `inference_dreambooth_lora.py`: code of model utilizing LoRA in DreamBooth.

*  `inference_dreambooth.py`: code fine-tuning simply use DreamBooth method.

*  `inference_lora.py`: code fine-tuning simply use LoRA method.

*  `inference_simple.py`: code fine-tuning without any technique.



### Configuration File

* `config_train.json` stores the parameters you need to change during training. 

* `config_test.json` stores the parameters you need to change during testing. 

### Utils

* Contains some tools to train and evaluate.




For any questions, please CONTACT Huihan Yang ASAP!
