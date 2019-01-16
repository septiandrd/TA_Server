import numpy as np
import cv2

size = '180 × 220'
ratio = '9 : 11'

eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def crop(img_path):
    img = cv2.imread(img_path)

    if img.shape[0] > 1000:
        scale_percent = 100/(img.shape[0]/500)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    if img.shape[1] > 1000:
        scale_percent = 100/(img.shape[1]/500)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces != ():
        for (x, y, w, h) in faces[:1]:
            y -= (h//10)
            if (x > 50) and (y > 50):
                face = img[y-50:y+h+50, x-50:x+w+50]
            elif (x > 30) and (y > 30):
                face = img[y-30:y+h+30, x-30:x+w+30]
            elif (x > 20) and (y > 20):
                face = img[y-20:y+h+20, x-20:x+w+20]
            elif (x > 10) and (y > 10):
                face = img[y-10:y+h+10, x-10:x+w+10]
            else:
                face = img[y:y+h, x:x+w]

        faceSize = face.shape[:2]
        grow = int(((11 / 9 * faceSize[0]) - faceSize[0]) / 2)

        img = cv2.copyMakeBorder(face, grow, grow, 0, 0, cv2.BORDER_REPLICATE)
        imsize = img.shape[:2]

        scale_percent = 100/(imsize[0]/220)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        img_name = '.'+img_path.split('.')[-2]+'-cropped.jpg'
        cv2.imwrite(img_name, img)

        return img_name
    else:
        return None


# img_path = './storage/S__8765447.jpg'
# img_path = './storage/410393_1.jpg'
# img_path = './storage/Zayn-Malik-inline.jpg'
# img_path = './storage/zayn-malik-t.jpg'
# img_path = './storage/NONFORMAL.jpg'
# img_path = './storage/bear.jpg'
# img_path = './storage/ecb3792f-4618-460e-bd84-6441085639ed.jpg'
# img_path = './storage/000007.jpg'
# crop(img_path)
