# README!!!

## Trainning
1. `config.json` stores the parameter you need to change.
For example, if you want to change the pre-trained model or the dataset, just change the content in `config.json`

2. to run the training please type in the following command in terminal
`cd diffusers/examples/text_to_image`
`python train_text_to_image.py` or i run the lora version for its effiency.

## Inferencing
1. `config_test.json` stores the parameter you need to change. PLEASE make sure that the parameter output_dir and pretrained_model_name_or_path is the SAME as the parameter output_dir and pretrained_model_name_or_path in `config.json`. To change the prompts, please change the content in prompts. parameter num_images refers to the number of pictures generated of each prompt.

2. to run the testing please type in the following command in terminal
`python inference.py`