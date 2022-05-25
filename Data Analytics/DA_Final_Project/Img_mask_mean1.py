import numpy as np
import PIL.Image
import os
import shutil
from tqdm import tqdm  # For nice progress bar!

# img_dir = os.getcwd() +'/train/images'

# i =0
# for file in os.listdir(img_dir):
#     if file.endswith(".jpg"):
#         print(os.path.join(img_dir, file))
#         # pass
#     i+=1
#     if i ==5:
#         break


        
class image_mean_cluster:
    def __init__(self, img_dir):
        # self.X = X
        self.img_dir = img_dir
    
    def cluster(self):
        # shutil.rmtree('newDatasets')
        # # os.removedirs('newDatasets')
        # os.mkdir('newDatasets')
        # os.chdir('newDatasets')
        k = 0
        for root, dirs, files in os.walk(self.img_dir):
            for file in tqdm(files):
                if file.endswith(".jpg"):
                    # print(root,dirs,file)
                    img_path = os.path.join(root, file)
                    img = np.array(PIL.Image.open(img_path))
                    for i in range(3):
                        mask = np.array([img[:,:,i].mean()>img[:,:,i] ])[0]

                        img[:,:,i][mask] = 255
                        # make it black
                        img[:,:,i][~mask] = 0
                    img1 = PIL.Image.fromarray(img)
                    img1.save(img_path)
                    # print(f'save {file} successfully')
                    # k+=1
                    # if k ==10:
                    #     return
        
if __name__ == "__main__":

    img_dir = '/home/jack/Document/yolov5/datasets/tissues_HVP-1/'
    # img_dir = '/home/jack/Document/yolov5/datasets/tissues_HVP-1/train/images'
    pre_work = image_mean_cluster(img_dir)
    pre_work.cluster()