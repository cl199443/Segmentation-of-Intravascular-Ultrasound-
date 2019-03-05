import numpy as np
import cv2
from numba import jit
from skimage import data,filters,segmentation,measure,morphology,color,feature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def bridge(con, image):
    if cv2.pointPolygonTest(con, (0, 0), False) >= 0 or cv2.pointPolygonTest(con, (0, image.shape[0]-1), False) >= 0 \
            or cv2.pointPolygonTest(con, (image.shape[1]-1, image.shape[0]-1), False) >= 0 or \
            cv2.pointPolygonTest(con, (image.shape[1]-1, 0), False) >= 0:
        return 1
    else:
        return 0

def minVal(img):
    return img.shape[0] * img.shape[1] / 50

def maxVal(img):
    return img.shape[0] * img.shape[1] / 3

def connection(image, thresh, cont_ans, regions, images):
    # #segmentation.clear_border(bw) #清除与边界相连的目标物
    # labels =measure.label(bw) #默认8连通标记
    # dst=color.label2rgb(labels) #根据不同的标记显示不同的颜色
    #删除小块区域 将面积小于300的小块区域删除（二值图像中由1变为0）
    #dst=morphology.remove_small_objects(data,min_size=300,connectivity=1)
    # thresh = filters.threshold_otsu(image)  # 阈值分割

    bw = morphology.closing(image < thresh, morphology.square(3))  # 闭运算:先膨胀再腐蚀 （bw 元素 : true / false）
    # plt.figure(), plt.imshow(image < thresh, 'gray'), plt.title(str(thresh))
    # plt.figure(), plt.imshow(bw, 'gray'), plt.title(str(thresh))
    # plt.show()
    # bw = (1-bw).astype(np.bool)
    cleared = bw.copy()  # 复制
    segmentation.clear_border(cleared)  # 清除与边界相连的目标物

    label_image = measure.label(cleared, connectivity=1)  # 连通区域标记
    # plt.imshow(label_image)
    # plt.show()
    # image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示
    # contours = measure.find_contours(cleared, 0.5)

    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
    # ax0.imshow(bw, plt.cm.gray, interpolation='nearest')
    # ax1.imshow(image_label_overlay, 'gray', interpolation='nearest')
    # plt.axis('off')
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 8))
    # ax0.imshow(cleared, plt.cm.gray)
    # ax1.imshow(image, plt.cm.gray)
    # for n, contour in enumerate(contours):
    #     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # ax0.set_title('cleared')
    # ax1.set_title('label_measure')
    # plt.show()

    for i, region in enumerate(measure.regionprops(label_image)):
        if region.area < minVal(image) or region.area > maxVal(image):  # add MGM later
            continue

        coor = region.coords  # (row, col)
        pixel = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        binary = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        for item in coor:  # item = (row, col)
            binary[item[0]][item[1]] = 255
            pixel[item[0]][item[1]] = image[item[0]][item[1]]

        # plt.imshow(binary, 'gray')
        # plt.show()
        d1, con, d = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # cv2.RETR_EXTERNAL : only save the biggest contour
        for k in range(len(con)):
            if cv2.contourArea(con[k]) < minVal(image) or cv2.contourArea(con[k]) > maxVal(image) or bridge(con[k], image):
                continue
            print(cv2.contourArea(con[k]))

            # plt.imshow(binary, 'gray')
            # plt.title("binary")
            # plt.show()

            ellipse = cv2.fitEllipse(con[k])
            temp = image.copy()
            cv2.ellipse(temp, ellipse, (255, 255, 0), 1)

            temp2 = image.copy()
            cv2.drawContours(temp2, con[k], -1, (255,255,255), 1)
            # plt.figure(), plt.imshow(temp, 'gray')
            # plt.figure(), plt.imshow(temp2, 'gray')
            # plt.show()

            # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 6))
            # ax0.imshow(binary, plt.cm.gray)
            # ax1.imshow(temp, plt.cm.gray)
            # ax1.plot(con[k][:, 0, 0], con[k][:, 0, 1], linewidth=1)
            # ax0.set_title('Measure_contours')
            # # plt.imshow(temp, 'gray')
            # plt.show()



            dist = np.zeros((image.shape[0], image.shape[1]))
            imgpix = np.zeros((image.shape[0], image.shape[1]))
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    dist[i][j] = cv2.pointPolygonTest(con[k], (j, i), False)

            local = np.argwhere(dist >= 0)
            for m in range(len(local)):
                imgpix[local[m][0]][local[m][1]] = image[local[m][0]][local[m][1]]
            images.append(imgpix)
            regions.append(local)
            cont_ans.append(con[k])