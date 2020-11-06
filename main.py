from PIL import Image, ImageDraw
from tkinter.filedialog import asksaveasfile
from tkinter.filedialog import askopenfilename
import os
import cv2
import random
import time
import numpy as np
import statistics as stat

def file_open_general():
    f = askopenfilename(title="Select file", filetypes=(("jpg files", "*.jpg"), ("png files", "*.png"),
                                                        ("jpeg files", "*.jpeg")))
    if f:
        return Image.open(f)


def file_open_cv():
    f = askopenfilename(title="Select file", filetypes=(("jpg files", "*.jpg"), ("png files", "*.png"),
                                                        ("jpeg files", "*.jpeg")))
    if f:
        return cv2.imread(f, cv2.IMREAD_COLOR)


def file_save(im):
    f = asksaveasfile(mode='w', defaultextension=".png")
    if f:  # asksaveasfile return `None` if dialog closed with "cancel".
        abs_path = os.path.abspath(f.name)
        im.save(abs_path)
    f.close()

def gray_scale_filter_pil():
    image = file_open_general()
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    for i in range(width):
        for j in range(height):
            r = pix[i, j][0]
            g = pix[i, j][1]
            b = pix[i, j][2]
            s = round(0.2126 * r + 0.7152 * g + 0.0722 * b)
            draw.point((i, j), (s, s, s))
    image.show()
    file_save(image)
    return image
def noise1(image):
    wigth = image.size[0]
    heigth = image.size[1]
    image1 = Image.new("RGB", (wigth, heigth))
    pix = image1.load()
    pix1 = image.load()
    for x in range(image1.size[0]):
        for y in range(image1.size[1]):
            rand = random.randint(-180, 180)
            rand = rand//2
            ImageDraw.Draw(image1).point((x, y), (rand, rand, rand))
    image1.show()
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            r1 = pix[x, y][0]
            g1 = pix[x, y][1]
            b1 = pix[x, y][2]
            r = pix1[x, y][0]
            g = pix1[x, y][1]
            b = pix1[x, y][2]
            r += r1
            g += g1
            b += b1
            if (r < 0):
                r = 0
            if (g < 0):
                g = 0
            if (b < 0):
                b = 0
            if (r > 255):
                r = 255
            if (g > 255):
                g = 255
            if (b > 255):
                b = 255
            ImageDraw.Draw(image).point((x, y), (r, g, b))
    image.show()
    file_save(image)
    return image

def center_point(image):
    image = image.convert('HSV')
    for x in range(1, image.size[0]-1):
        for y in range(1, image.size[1]-1):
            hsv = image.load()
            H = hsv[x, y][0]
            S = hsv[x, y][1]
            V = hsv[x, y][2]
            min1 = 1000
            max1 = 0
            for i in range(x-1,x+1):
                for j in range(y-1,y+1):
                        s1 = hsv[i, j][2]
                        if (s1 > max1):
                            max1 = s1
                        if (s1 < min1):
                            min1 = s1
            V = (max1 + min1) // 2
            ImageDraw.Draw(image).point((x, y), (H, S, V))
    image = image.convert('RGB')
    image.show()


def noise2(image):
    image = image.convert('HSV')
    pix = image.load()
    #pix1 = image.load()
    for x in range(image.size[0]):
        for y in range(image.size[1]):
            rand = random.randint(5, 80)
            if rand == 80:
                V = random.randint(1, 100)
            else:
                V = pix[x, y][2]
            ImageDraw.Draw(image).point((x, y), (pix[x,y][0], pix[x,y][1], V))
    image = image.convert('RGB')
    image.show()
    file_save(image)
    return image


def brightHSV(img1, value):
    imgHSV = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(imgHSV)
    v += value
    img = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2BGR)
    return img

def harmonic(image):
    image = image.convert('HSV')
    wigth = image.size[0]
    heigth = image.size[1]
    image1 = Image.new("HSV", (wigth, heigth))
    hsv = image.load()
    for x in range(2, image.size[0]-1):
        for y in range(2, image.size[1]-1):
            H = hsv[x, y][0]
            S = hsv[x, y][1]
            V = hsv[x, y][2]
            sum0 = 0
            sum1 = 0
            sum11 = 0
            for i in range(x-2, x+2):
                for j in range(y-2, y+2):
                    if hsv[i, j][0] != 0:
                        sum0 += 1/hsv[i, j][0]
                    if hsv[i, j][1] != 0:
                        sum1 += 1/hsv[i, j][1]
                    if hsv[i, j][2] != 0:
                        sum11 += 1/hsv[i, j][2]
            sum0 = sum0 / 9
            sum1 = sum1 / 9
            sum11 = sum11 / 9
            if sum0 != 0:
                H = (round)(1/sum0)
            if sum1 != 0:
                S = (round)(1/sum1)
            if sum11 != 0:
                V = (round)(1/sum11)
            ImageDraw.Draw(image1).point((x, y), (H, S, V))
    image1 = image1.convert('BGR')
    img = brightHSV(image1, 10)
    image = Image.fromarray(img)
    image1.show()
    file_save(image1)
    return image1

def opencv_grsc_denoise():
    cv2.startWindowThread()
    image = file_open_cv()
    #blank_image = np.zeros((image.shape), np.uint8)
    result = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    cv2.namedWindow('Press esc to close')
    cv2.imshow('Press esc to close', result)
    gray_image = Image.fromarray(result)
    file_save(gray_image)
    # cv2.waitKey()
    cv2.destroyAllWindows()



print("Choose preferred mode:\n"
      "1 - Noise addition filter + Denoising using Center point filter\n"
      "2 - Denoising using Center point filter\n"
      "3 - Denoising using Harmonic mean filter\n"
      "4 - Denoising using fastNlMeansDenoising filter(OpenCV) for grayscale images\n")
print('Input mode № = ')
mode = input()
if mode == '1':
     img = Image.open('C:/Users/Kate/Pictures/filters/image6.png')
     d = noise1(img)
elif mode == '2':
     img_noise = Image.open('C:/Users/Kate/Pictures/filter_results/image_noised_.png')
     d = center_point(img_noise)
elif mode == '3':
    img_noise = Image.open('C:/Users/Kate/Pictures/filter_results/image_noised_.png')
    d = harmonic(img_noise)
elif mode == '4':
    opencv_grsc_denoise()





#img_noise = Image.open('C:/Users/Kate/Pictures/index.jpeg')
#d = center_point(img_noise)
"""def opencv_denoise():
        cv2.startWindowThread()
        image = file_open_cv()
        # blank_image = np.zeros((image.shape), np.uint8)
        result = cv2.fastNlMeansDenoisingColored(image, None, 20, 10, 21, 7)
        cv2.namedWindow('Press esc to close')
        cv2.imshow('Press esc to close', result)
        gray_image = Image.fromarray(result)
        file_save(gray_image)
        # cv2.waitKey()
        cv2.destroyAllWindows()"""

#img_noise = noise2(Image.open('C:/Users/Kate/Pictures/filters/image6.png'))