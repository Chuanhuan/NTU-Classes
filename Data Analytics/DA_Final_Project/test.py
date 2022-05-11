
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
img_file = os.getcwd()+'/supe_0067.jpg'
img = mpimg.imread(img_file)
plt.imshow(img)
img.shape
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


from PIL import Image, ImageDraw
import numpy as np

cwd = os.getcwd()
img_file = os.getcwd()+'/supe_0067.jpg'
img = mpimg.imread(img_file)
img.shape
x = img[0]+img[1]+img[2]
x = np.matrix(x).sum(axis=1)
np.where(x<300)[0]
img.convert('L')


img1 = Image.open(img_file)
img1 = img1.convert("L")
img1.show()
plt.imshow(img1)
extrema = img1.getextrema()
np.array(img1)

for i in img

x = 10
r = 20
image = Image.open("x.png")
draw = ImageDraw.Draw(image)
leftUpPoint = (x-r, y-r)
rightDownPoint = (x+r, y+r)
twoPointList = [leftUpPoint, rightDownPoint]
draw.ellipse(twoPointList, fill=(255,0,0,255))
# %%


  
# creating image object
img = Image.open(img_file)
  
# using convert method for img1
img1 = img.convert("L")
img1.show()
  
# using convert method for img2
 img2 = img.convert("1")
img2.show()