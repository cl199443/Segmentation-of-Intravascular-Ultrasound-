import numpy as np
import cv2
import matplotlib.pyplot as plt
# import mser
from PIL import Image

#contours是提取的所有的轮廓，如contours[0]、contours[1]...
#regions是每个轮廓围成的区域内各点像素值的坐标，类似mser返回的坐标点
#img为每个轮廓内区域的原像素点,imgs[0]、imgs[1]...

def toSolveAmedian(contours):
    arealist=[]
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # contours[i].area
        arealist.append(area)
    sort_area = np.sort(arealist)
    Amedian = sort_area[len(sort_area)//2] # if len(sort_area)%2==1 else (sort_area[(len(sort_area)//2) - 1]+sort_area[len(sort_area)//2])/2
    return Amedian

def toSolveMAD(contours,Amedian):
    MADlist = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # contours[i].area
        MADs=abs(area-Amedian)###################################改动###########################################
        MADlist.append(MADs)
    sort_MAD=np.sort(MADlist)
    MAD=sort_MAD[len(MADlist)//2]
    return MAD+0.01

def toSolveMi(contours,Amedian,MAD):
    Mi=[]
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])  # contours[i].area
        Mi.append(0.6745*(area-Amedian)/MAD)

    assert len(contours) == len(Mi)###################################改动###########################################
    return Mi

def Z_selection(Mi,contours,regions,imgs):
    index = [i for i, item in enumerate(Mi) if abs(item) >= 3]
    Mi = [item for i, item in enumerate(Mi) if i not in index]
    contours = [item for i, item in enumerate(contours) if i not in index]
    regions = [item for i, item in enumerate(regions) if i not in index]

    return Mi, contours, regions, imgs   #regions和imgs中也要相应的删除，即该返回值应有Mi、contours、regions、imgs


def toSolveV(contours,regions,imgs):
    V=[]
    for i in range(len(contours)):
        L = cv2.arcLength(contours[i], True)  # contours[i].perimeter

        E, Etotal = 0, 0   ###################################改动###########################################
        pixelArea = []  #存储区域块内每一个像素点的像素值
        for j in range(len(regions[i])):
            Etotal += imgs[i][regions[i][j][0]][regions[i][j][1]]
            pixelArea.append(imgs[i][regions[i][j][0]][regions[i][j][1]])

        N = cv2.contourArea(contours[i])  # contours[i].area
        E = Etotal / N

        # hist = np.histogram(imgs[i], bins=256)  ###################################改动###########################################
        pixelArea = np.array(pixelArea)
        # print(pixelArea)
        hist = np.histogram(pixelArea, bins=int((pixelArea.max() - pixelArea.min() + 1)))
        count = hist[0]  #是个数组
        #gray_value=hist[1]   #是个数组
        H=0
        total = count.sum()
        # assert total == N
        for k in range(len(count)):
            p=count[k]/total
            if p!=0:
                logp=np.log2(p)
                entropele=-p*logp
                H=H+entropele

        V.append(L*E*H)
    return V
def toSolveOmiga(V):
    Omiga = []
    for i in range(len(V)):
        if i==0 or i==len(V)-1:   #???   i==len(V)-1就报错    ???
            Omiga.append(0)
        else:
            value=V[i+1]-V[i-1] + 0.001  ###################################改动###########################################
            Omiga.append(V[i]/value)

    assert len(V) == len(Omiga)
    return Omiga


def draw_img(Omiga):
    x=[]
    #i也代表第i个区域，找出局部最大值之后方便找轮廓
    for i in range(len(Omiga)):
        x.append(i)

    plt.plot(x, Omiga)
    plt.xlabel('Q+Reggions')
    plt.ylabel('stability score(Omiga)')
    plt.axis('tight')
    plt.show()


def local_max(Omiga):
    maxlist=[]
    target_index=[]
    for i in range(1,len(Omiga)-1):
        if Omiga[i]>=max(Omiga[i-1],Omiga[i+1]):
            maxlist.append(Omiga[i])
            target_index=i

    return maxlist,target_index   #选择哪个local_max,就对应contours[target_index]


# if __name__ == '__main__':
#     img = np.array(Image.open('test.bmp').convert('L'))
#     contours, regions, imgs = mser.mser(img)
#
#     Amedian = toSolveAmedian(contours)
#     # print('Amedian:',Amedian)
#
#     MAD=toSolveMAD(contours,Amedian)
#     # print('MAD:',MAD)
#
#     Mi=toSolveMi(contours, Amedian, MAD)
#     # print('Mi:',Mi)
#     #
#     # M,contours,regions,imgs=Z_selection(Mi,contours,regions,imgs)
#     # print(M)
#
#     V=toSolveV(contours, regions, imgs)
#     # print(V)
#
#     Omiga=toSolveOmiga(V)
#     # print(Omiga)
#
#     draw_img(Omiga)