from PIL import Image
import os

def resize():
    original_folder_path = 'pics/real_images'
    save_folder_path = 'pics/resized_original_pic'
    original_pic = os.listdir(original_folder_path)
    for i in original_pic:
        image_path = os.path.join(original_folder_path,i)   

        # 定义目标大小
        target_size = (512, 512)

        # 打开图像
        image = Image.open(image_path)

        # 调整大小
        resized_image = image.resize(target_size, resample=Image.BILINEAR)
        saved_path = os.path.join(save_folder_path,i)
        resized_image.save(saved_path)

if __name__ == "__main__":
    resize()