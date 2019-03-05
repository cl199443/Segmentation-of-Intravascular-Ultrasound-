from numpy import *
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from EREL import *

#SEG, GT are the binary segmentation and ground truth areas, respectively



def Dice_Ratio(seg,gt):
    he1=seg & gt
    return 2 * np.sum(he1) / (np.sum(seg)+np.sum(gt))

def Jaccard_ratio(seg,gt):
       he2=seg&gt
       up = np.sum(he2)
       he3=seg|gt
       down = np.sum(he3)
       return up/down


def MaxMinDist(lu, me):
    minDist = []
    for i in range(lu.shape[0]):
        temp = lu[i] - me
        dist = [np.sqrt(item[0]**2+item[1]**2) for item in temp]
        assert len(dist) == me.shape[0]
        minDist.append(min(dist))
    return max(minDist)

def HD(seg_media1,gt_media, img):

    d1, cor_lu, d3 = cv2.findContours(seg_media1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    d2, cor_me, d4 = cv2.findContours(gt_media, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # ellipse = cv2.fitEllipse(cor_lu[0])
    # ellipse1 = cv2.fitEllipse(cor_me[0])
    # temp = img.copy()
    # cv2.ellipse(temp, ellipse, (0, 255, 0), 1)  # the fourth parameter is the line-width
    # cv2.ellipse(temp, ellipse1, (255, 0, 0), 1)  # the fourth parameter is the line-width
    # plt.imshow(temp, 'gray')
    # plt.show()

    cor_lu, cor_me = np.array(cor_lu[0]), np.array(cor_me[0])
    cor_lu = cor_lu.reshape(cor_lu.shape[0], cor_lu.shape[2])
    cor_me = cor_me.reshape(cor_me.shape[0], cor_me.shape[2])
    assert cor_lu.shape[1] == cor_me.shape[1]

    sg = MaxMinDist(cor_lu, cor_me)
    gs = MaxMinDist(cor_me, cor_lu)
    return max(sg, gs) / 37.8

def PAD(seg,gt):
    # print("sum(seg), sum(gt) is :", sum(seg), sum(gt))
    return abs(int(sum(seg))-int(sum(gt)))/sum(gt)  # change the type of uint32 into int


def getbinary(img, path):
    print(path)
    cur = 'dataset\Training_Set\Data_set_B\LABELS/'
    data_lu=[]
    for line in open(cur + 'lum_'+ path + '.txt'):
        line=line.strip("\n")
        line=line.split(',')
        arr1=list(map(float,line))
        # print(arr)
        data_lu.append(arr1)
    data_lu=np.array(data_lu,np.float32)

    data_me=[]
    for line in open(cur + 'med_'+ path + '.txt'):
        line=line.strip("\n")
        line=line.split(',')
        arr2=list(map(float,line))
        # print(arr)
        data_me.append(arr2)
    data_me=np.array(data_me,np.float32)

    gt_lumen = np.zeros((img.shape[0], img.shape[1]),np.uint8)
    gt_media = np.zeros((img.shape[0], img.shape[1]),np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if cv2.pointPolygonTest(data_lu, (j, i), False) >= 0:
                gt_lumen[i][j] = 1
            if cv2.pointPolygonTest(data_me, (j, i), False) >= 0:
                gt_media[i][j] = 1

    # d1, con1, r1 = cv2.findContours(gt_lumen, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # temp1 = img.copy()
    # cv2.drawContours(temp1,con1,-1,(255,255,255),1)
    #
    # d2, con2, r2 = cv2.findContours(gt_media, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # temp2 = img.copy()
    # cv2.drawContours(temp2,con2,-1,(255,255,255),1)
    # plt.figure(1),plt.title("truth")
    # plt.imshow(temp1,'gray')
    #
    # plt.figure(2),plt.title("truth")
    # plt.imshow(temp2,'gray')
    # plt.show()

    imgFilter = filter(img)
    MGM, vx = toSolveMGM(imgFilter, img, 0.5)  # MGM:array with size (height, width)
    MGMplus, MGMminus = toSeparateMGM(MGM, img, vx)  # MGMplus and MGMminus:array with size (height, width)

    VvList = []  # 256 * 2 (plus, minus)
    for level in range(0, 256):  # level is between [0,255]
        Tplus, Tminus = toSolveTv(img, level)
        Nvplus, Nvminus = toSolveNv(Tplus, MGMplus), toSolveNv(Tminus, MGMminus)
        Vvplus, Vvminus = toSolveVv(Nvplus, Tplus), toSolveVv(Nvminus, Tminus)
        VvList.append([Vvplus, Vvminus])
    Fai = toSolveFai(VvList)
    ELplus, ELminus = toSolveEL(Fai)

    contours, regions, imgs = [], [], []
    for level in ELminus:
        if level < 1.5 * vx and level > 12:
            connection(img, level, contours, regions, imgs)
        else:
            continue

    V = toSolveV(contours, regions, imgs)

    Omiga = toSolveOmiga(V)
    # print("Omiga is : {} and the size is : {}".format(Omiga,len(Omiga)))

    # draw_img(Omiga)

    assert len(Omiga) == len(contours)

    ans = []
    for i in range(1, len(Omiga)-1):
        if Omiga[i] > Omiga[i-1] and Omiga[i] > Omiga[i+1]:
            ans.append(contours[i])

    seg_lumen = np.zeros((img.shape[0], img.shape[1]),np.uint8)
    seg_media = np.zeros((img.shape[0], img.shape[1]),np.uint8)


    ellipse_lu = cv2.fitEllipse(ans[1])
    ellipse_me = cv2.fitEllipse(ans[len(ans)-3])

    cv2.ellipse(seg_lumen, ellipse_lu, (255, 255, 255), 1)
    cv2.ellipse(seg_media, ellipse_me, (255, 255, 255), 1)
    cor_lu=np.argwhere(seg_lumen==255)
    cor_me=np.argwhere(seg_media==255)

    for i in range(cor_lu.shape[0]):
        temp_lu=cor_lu[i][0]
        cor_lu[i][0]=cor_lu[i][1]
        cor_lu[i][1]=temp_lu
    for i in range(cor_me.shape[0]):
        temp_me=cor_me[i][0]
        cor_me[i][0]=cor_me[i][1]
        cor_me[i][1]=temp_me

    # corlu = cv2.fitEllipse(cor_lu)
    # cv2.ellipse(seg_lumen, corlu, (255, 255, 255), 1)
    # plt.figure(), plt.title("forelumedia")
    # plt.imshow(seg_lumen)
    # plt.show()


    seg_lumen1 = np.zeros((img.shape[0], img.shape[1]),np.uint8)
    seg_media1 = np.zeros((img.shape[0], img.shape[1]),np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if cv2.pointPolygonTest(cor_lu, (j, i), False) >= 0:
                seg_lumen1[i][j] = 1
            if cv2.pointPolygonTest(cor_me, (j, i), False) >= 0:
                seg_media1[i][j] = 1
    return seg_lumen1,seg_media1,gt_lumen,gt_media


if __name__ == '__main__':

    filenameLU, filenameMED = 'resultLU.txt',  'resultMED.txt'
    with open(filenameLU, 'a') as f:  # the second img
        f.write('DICE' + ' ,' + 'JA' + ' ,' + 'HD' + ' ,' + 'PAD' + '\n')
    with open(filenameMED, 'a') as f:  # the second img
        f.write('DICE' + ' ,' + 'JA' + ' ,' + 'HD' + ' ,' + 'PAD' + '\n')

    for i in range(1, 3):
        for j in range(1, 51):
            flag = '_000' + str(j) if j < 10 else '_00' + str(j)
            path = 'dataset\Training_Set\Data_set_B\DCM/'
            name = 'frame_0' + str(i) + flag + '_003'

            img = np.array(Image.open(path + name + '.png').convert('L'))
            seg_lumen1, seg_media1, gt_lumen, gt_media = getbinary(img, name)

            dice = Dice_Ratio(seg_lumen1, gt_lumen)
            ja = Jaccard_ratio(seg_lumen1, gt_lumen)
            hd = HD(seg_lumen1, gt_lumen, img)  # HD(seg_media1,gt_media)
            pad = PAD(seg_lumen1, gt_lumen)

            dice1 = Dice_Ratio(seg_media1, gt_media)
            ja1 = Jaccard_ratio(seg_media1, gt_media)
            hd1 = HD(seg_media1, gt_media, img)# HD(seg_media1,gt_media)
            pad1 = PAD(seg_media1, gt_media)

            with open(filenameLU, 'a') as f:  # the second img
                f.write(str(dice) + ' ,' + str(ja) + ' ,' + str(hd) + ' ,' + str(pad) + '\n')
            with open(filenameMED, 'a') as f:  # the second img
                f.write(str(dice1) + ' ,' + str(ja1) + ' ,' + str(hd1) + ' ,' + str(pad1) + '\n')

            # print(dice)
            # print(ja)
            # print(hd)
            # print(pad)

            # d3, con3, r3 = cv2.findContours(seg_lumen1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # temp3 = img.copy()
            # cv2.drawContours(temp3,con3,-1,(255,255,255),1)
            #
            # d4, con4, r4 = cv2.findContours(seg_media1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            # temp4 = img.copy()
            # cv2.drawContours(temp4,con4,-1,(255,255,255),1)

            # plt.figure(1)
            # plt.title('our seg lumen')
            # plt.imshow(seg_lumen1,'gray')
            # plt.figure(2)
            # plt.title('our seg media')
            # plt.imshow(seg_media1,'gray')
            # plt.figure(3)
            # plt.title('standard lumen')
            # plt.imshow(gt_lumen,'gray')
            # plt.figure(4)
            # plt.title('standard media')
            # plt.imshow(gt_media,'gray')
            # plt.show()














