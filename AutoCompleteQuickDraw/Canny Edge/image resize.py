from PIL import Image

from resizeimage import resizeimage
import numpy as np

def resize(src, dest):
    with open(src, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [200, 200])
            cover.save(dest, image.format)

def convertpixels(src):
    i = Image.open(src)
    
    pixels = i.load() # this is not a list, nor is it list()'able
    width, height = i.size

    all_pixels = []
    for x in range(width):
        for y in range(height):
            cpixel = pixels[x, y]
            all_pixels.append(cpixel)
    print(all_pixels[100:500])

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
    

def main():
    a=mse(Image.open("0.png"),Image.open("house2.png"))
    print(a)
    
if __name__=="__main__":
    main()
