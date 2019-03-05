import numpy as np
import matplotlib.pyplot as plt
import cv2
from numba import jit
from PIL import Image
from connection import *
from selection import *
from ROFfilter import *
from numba import jit
from skimage import data,filters,segmentation,measure,morphology,color

def filter(img):  # the first step to do filtering
    scharrx = cv2.Scharr(img, cv2.CV_64F, dx=1, dy=0)
    scharrx = cv2.convertScaleAbs(scharrx)
    scharry = cv2.Scharr(img, cv2.CV_64F, dx=0, dy=1)
    scharry = cv2.convertScaleAbs(scharry)
    result = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
    return result

def toSolveMGM(imgFilter, img, alpha):
    MGM = np.zeros(imgFilter.shape)
    imgPad = np.pad(imgFilter, 2, 'constant')
    valMean = np.mean(imgFilter)
    MGMList = []
    for i in range(imgFilter.shape[0]):
        for j in range(imgFilter.shape[1]):
            meanPatch = np.mean(imgPad[i:i+5, j:j+5])
            maxPatch = np.max(imgPad[i:i+5, j:j+5])
            MGM[i][j] = 1 if meanPatch >= alpha*valMean and imgFilter[i][j] >= maxPatch else 0
            if(MGM[i][j]):
                MGMList.append(img[i][j])
    MGMList = np.sort(MGMList)
    return MGM, MGMList[len(MGMList)//2]

def toSeparateMGM(MGM, img, vx):
    MGMplus, MGMminus = np.zeros(img.shape), np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if MGM[i][j] == 1:
                MGMplus[i][j] = 1 if img[i][j] >= vx else 0
                MGMminus[i][j] = 1 if img[i][j] >= (256-vx) else 0
    return MGMplus, MGMminus

@jit
def toSolveTv(img, level):
    Tplus, Tminus = np.zeros(img.shape), np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            Tplus[i][j] = 1 if img[i][j] >= level else 0
            Tminus[i][j] = 1 if img[i][j] <= level else 0

    return Tplus, Tminus

@jit
def toSolveNv(Tv, MGM):
    assert Tv.shape == MGM.shape

    temp = Tv * MGM
    return np.sum(temp)

@jit
def toSolveVv(Nv, Tv): # Nv is constant int type, Tv is array type

    temp = Nv/(1 + np.sum(Tv))
    return temp

@jit
def toSolveFai(Vv):
    Fai = []
    for i in range(len(Vv)):
        if i == 0 or i == 255:
            Fai.append([0, 0])
        else:
            plus = Vv[i][0]/(0.01 + abs(Vv[i-1][0]-Vv[i+1][0]))
            minus = Vv[i][1]/(0.01 + abs(Vv[i-1][1]-Vv[i+1][1]))
            Fai.append([plus, minus])

    return Fai

@jit
def toSolveEL(Fai):
    assert len(Fai) == 256

    ELplus, ELminus = [], []
    for level in range(1, 255):  # delete the 0 and 255 level
        if Fai[level][0] >= max(Fai[level-1][0], Fai[level+1][0]):
            ELplus.append(level)
        if Fai[level][1] >= max(Fai[level-1][1], Fai[level+1][1]):
            ELminus.append(level)

    return ELplus, ELminus

if __name__ == '__main__':
    img = np.array(Image.open('frame_01_0035_003.png').convert('L'))  # ROF(np.array(Image.open('001.png').convert('L')))
    plt.imshow(img, 'gray')
    plt.show()
    imgFilter = filter(img)  # scharr 算子 as filter
    # plt.figure(), plt.imshow(img, 'gray')
    # plt.figure(), plt.imshow(imgFilter, 'gray')
    # plt.show()

    MGM, vx = toSolveMGM(imgFilter, img, 0.5)  # MGM:array with size (height, width)
    print(vx)
    # plt.imshow(MGM, 'gray')
    # plt.show()
    MGMplus, MGMminus = toSeparateMGM(MGM, img, vx)  # MGMplus and MGMminus:array with size (height, width)

    VvList = []  # 256 * 2 (plus, minus)
    for level in range(0, 256):  # level is between [0,255]
        Tplus, Tminus = toSolveTv(img, level)
        Nvplus, Nvminus = toSolveNv(Tplus, MGMplus), toSolveNv(Tminus, MGMminus)
        Vvplus, Vvminus = toSolveVv(Nvplus, Tplus), toSolveVv(Nvminus, Tminus)
        VvList.append([Vvplus, Vvminus])
    Fai = toSolveFai(VvList)

    ELplus, ELminus = toSolveEL(Fai)
    # EREL = [val for val in ELplus if val in ELminus]  # solve the intersection of ELplus, ELminus
    #
    # print(EREL)

    print("ELminus is :", ELminus)

    contours, regions, imgs = [], [], []
    # ELplus = [1, 4, 7, 13, 17, 19, 22, 26, 36, 40, 44, 51, 59, 66, 73, 77, 81, 88, 91, 96, 103, 108, 110, 114, 118, 125, 133, 140, 147, 155, 162, 170, 177, 181, 184, 189, 192, 199, 207, 214, 221, 225, 229, 232, 236, 241, 244, 250, 254]
    # for level in ELminus:
    #     plt.imshow(img < level, 'gray')
    #     plt.title(level)
    #     plt.show()

    for level in ELminus:
        if level < 1.5 * vx and level > 12:
            connection(img, level, contours, regions, imgs)
        else:
            continue

    # plt.imshow(img, 'gray')
    # for contour1 in contours:
    #     plt.plot(contour1[:, 0, 0], contour1[:, 0, 1], linewidth=2)
    # plt.show()

    Amedian = toSolveAmedian(contours)
    # print('Amedian:',Amedian)

    MAD = toSolveMAD(contours, Amedian)
    # print('MAD:',MAD)

    Mi = toSolveMi(contours, Amedian, MAD)
    # print('Mi:',Mi)
    #
    # M, contours, regions, imgs = Z_selection(Mi, contours, regions, imgs)
    # print(M)

    V = toSolveV(contours, regions, imgs)
    # print(V)

    Omiga = toSolveOmiga(V)
    print("Omiga is : {} and the size is : {}".format(Omiga,len(Omiga)))

    draw_img(Omiga)

    assert len(Omiga) == len(contours)

    # for i in range(0, len(Omiga)):
    #     plt.imshow(img, 'gray')
    #     plt.plot(contours[i][:, 0, 0], contours[i][:, 0, 1], linewidth=2)
    #     plt.axis('off')
    #     plt.show()
    ans = []
    for i in range(1, len(Omiga)-1):
        if Omiga[i] > Omiga[i-1] and Omiga[i] > Omiga[i+1]:
            ans.append(contours[i])

    print("the peak size is :", len(ans))
    for i in range(len(ans)):
        ellipse = cv2.fitEllipse(ans[i])
        temp = img.copy()
        cv2.ellipse(temp, ellipse, (255, 255, 255), 1)  # the fourth parameter is the line-width
        plt.imshow(temp, 'gray')
        plt.plot(ans[i][:, 0, 0], ans[i][:, 0, 1], linewidth=1)
        plt.axis('off')
        plt.title('local maxima peak of ' + str(i+1))
        plt.show()

    for i in range(3):
        flag = 0 if i == 0 else 1 if i == 1 else len(ans)-2
        title = 'lumen of ' + str(flag) if flag == 0 or flag == 1 else "media"
        ellipse = cv2.fitEllipse(ans[flag])
        temp = img.copy()
        cv2.ellipse(temp, ellipse, (255, 255, 0), 1)  # the fourth parameter is the line-width
        plt.imshow(temp, 'gray')
        plt.plot(ans[flag][:, 0, 0], ans[flag][:, 0, 1], linewidth=1)
        plt.axis('off')
        plt.title(title)
        plt.show()