
import PIL
from numpy import size
import torchvision.transforms.functional as transform
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms.functional as TF
import numpy as np

cwd = os.getcwd()
img_file = os.getcwd()+'/dysk100.png'
img = mpimg.imread(img_file)
plt.imshow(img)
# Reads a file using pillow
PIL_image = PIL.Image.open(img_file)

# The image can be converted to tensor using
tensor_image = transform.to_tensor(PIL_image)

tensor_image.size()

# The tensor can be converted back to PIL using
new_PIL_image = transform.to_pil_image(tensor_image)
# new_PIL_image.show()
# new_PIL_image.close()

img = cv2.imread(img_file)
res = cv2.resize(img, dsize=(10, 10), interpolation=cv2.INTER_CUBIC)

res.shape
plt.imshow(res)


res1= TF.resize(PIL.Image.open(img_file),size=[100,100])
mpimg.imread(res1)
np.array(res1).shape

# # explore transformation data by using matplotlib
# png_df.iloc[0,1].shape # (84, 72, 3)
# tt = png_df.iloc[0,1].reshape(-1,3)
# tt.shape
# tt = tt.reshape(84,72,3)
# tt.shape
# plt.imshow(png_df.iloc[0,1])
# plt.imshow(tt)



# # show image tensor format
# k = 0
# # insert all gender to new row
# for i in unique_cell:    
#     # The tensor can be converted back to PIL using
#     new_PIL_image = transform.to_pil_image(png_df.iloc[0,cell_list.index(i)])
#     new_PIL_image.show()
#     new_PIL_image.close()
#     k = k +1