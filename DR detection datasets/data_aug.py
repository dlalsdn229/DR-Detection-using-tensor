import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
#from keras.preprocessing.image import ImageDataGenerator

path = 'C:/Users/User/Downloads/DRIMDB/DRIMDB/Bad/'
d_path = "C:/Users/User/Downloads/DRIMDB/DRIMDB/aug/Bad/"
filesarray = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path,x))]

def flip(img):
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    lr_img = cv.flip(img_rgb,1)
    tb_img = cv.flip(img_rgb, 0)
    '''
    plt.subplot(1,2,1)
    plt.imshow(img_rgb)
    plt.subplot(1,2,2)
    plt.imshow(img_flip)

    plt.show()
    '''
    lr_img = cv.cvtColor(lr_img,cv.COLOR_RGB2BGR)
    tb_img = cv.cvtColor(tb_img, cv.COLOR_RGB2BGR)

    return lr_img,tb_img

def rotate(img,degree):
    height,width,channel = img.shape
    if height > width:
        height = width
    else:
        width = height
    img = cv.resize(img,dsize=(width,height))

    matrix = cv.getRotationMatrix2D((height/2,width/2),degree,1)
    result = cv.warpAffine(img, matrix, (height, width))
    #result = cv.warpAffine(img,matrix,(int(height*3/4),int(width*5/6)))

    return result

def brightness(img):
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    M = np.ones(img_rgb.shape, dtype = "uint8") * 100
    br_img = cv.add(img_rgb,M)
    m = np.ones(img_rgb.shape, dtype = "uint8") * 50
    dk_img = cv.subtract(img_rgb,m)
    '''
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.subplot(1, 2, 2)
    plt.imshow(dk_img)

    plt.show()
    '''
    br_img = cv.cvtColor(br_img, cv.COLOR_RGB2BGR)
    dk_img = cv.cvtColor(dk_img, cv.COLOR_RGB2BGR)

    return br_img, dk_img

def morph(img):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, threshold = cv.threshold(img_gray, 100, 255, cv.THRESH_BINARY)
    cv.imshow('threshold',threshold)

    erode = cv.erode(threshold,(7,7), iterations=1)
    cv.imshow('erode', erode)

    dilate = cv.dilate(erode,(7,7), iterations=1)
    cv.imshow('dilate',dilate)

    cv.waitKey(0)

def datagen(img):
    data_aug_gen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.5,zoom_range=[0.8,0.2],horizontal_flip=True,vertical_flip=True, fill_mode='nearest')


count = 0
for file_name in filesarray:
    count= count+1
    filename = os.path.splitext(file_name)[0]
    img = cv.imread(path + '/' + file_name)

    #상하좌우반전
    lr, tb = flip(img)
    r90 = rotate(img,90)
    t,r270 = flip(r90)
    #밝기
    #bright, dark = brightness(img)

    #morph(img)

    #cv.imwrite(d_path + filename + ".jpeg", img)
    cv.imwrite(d_path + filename + "lr.jpeg",lr)
    cv.imwrite(d_path + filename + "tb.jpeg", tb)
    cv.imwrite(d_path + filename + "r90.jpeg", r90)
    cv.imwrite(d_path + filename + "r270.jpeg", r270)
'''
    cv.imwrite(d_path + filename + "br.jpeg", bright)
    cv.imwrite(d_path + filename + "dk.jpeg", dark)
'''
    #print(str(count)+ '.' + filename + "사진 작업 완료")