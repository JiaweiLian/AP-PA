import cv2
import numpy as np

def Gaussnoise_func(image, mean=0, var=0.005):
    '''
    添加高斯噪声
    mean : 均值
    var : 方差
    '''
    image = np.array(image/255, dtype=float)                    #将像素值归一
    noise = np.random.normal(mean, var ** 0.5, image.shape)     #产生高斯噪声
    out = image + noise                                         #直接将归一化的图片与噪声相加

    '''
    将值限制在(-1/0,1)间，然后乘255恢复
    '''
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.

    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def nothing(pp):
    pass

if __name__ == '__main__':
    img = cv2.imread("/home/mnt/ljw305/adversarial-yolo/patches/center_150_1024_yolov2.png")
    #创建预览界面
    cv2.namedWindow("Preview")
    cv2.createTrackbar("mean","Preview",0,5,nothing)
    cv2.createTrackbar("var","Preview",0,5,nothing)
    while(1):
        mean = cv2.getTrackbarPos("mean","Preview")
        var = cv2.getTrackbarPos("var","Preview")
        img_r = Gaussnoise_func(img,mean/10,var/100)
        cv2.imshow("Result",img_r)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
    cv2.destroyAllWindows()