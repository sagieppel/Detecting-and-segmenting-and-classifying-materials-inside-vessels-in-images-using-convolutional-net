#############################################Reader for COCO related data for vessels for data generated in the COCO #######################################################

import cv2
import numpy as np
import random


#############################################################################################
def show(Im):
    cv2.imshow("show",Im.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
##############################################################################################

###############################################################################################


#Reader for the coco panoptic data set for pointer based image segmentation
import numpy as np
import os
#import scipy.misc as misc
import random
import cv2
import json
import threading
import random
############################################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir,AnnDir, MaxBatchSize=100,MinSize=250,MaxSize=1000,MaxPixels=800*800*5,TrainingMode=True):

        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.ImageDir = ImageDir
        self.AnnDir = AnnDir
        self.Epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.ClassBalance=False
# ----------------------------------------Create list of files--------------------------------------------------------------------------------------------------------------
        self.AnnList = [] # Image/annotation list

        print("Creating annotation list for reader this might take a while")
        for ann in os.listdir(AnnDir):
            if ".png" not in ann: continue
            self.AnnList.append(ann)
        if TrainingMode:
            np.random.shuffle(self.AnnList)

        print("Total=" + str(len(self.AnnList)))
        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize(self,Img, AnnMap,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================

        h,w,d=Img.shape
        Bs = np.min((h/Hb,w/Wb))
        if Bs<1 or Bs>1.5:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(h / Bs)+1
            w = int(w / Bs)+1
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            AnnMap = cv2.resize(AnnMap, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
 # =======================Crop image to fit batch size===================================================================================


        if np.random.rand()<0.9:
            if w>Wb:
                X0 = np.random.randint(w-Wb)
            else:
                X0 = 0
            if h>Hb:
                Y0 = np.random.randint(h-Hb)
            else:
                Y0 = 0

            Img=Img[Y0:Y0+Hb,X0:X0+Wb,:]
            AnnMap = AnnMap[Y0:Y0+Hb,X0:X0+Wb]

        if not (Img.shape[0]==Hb and Img.shape[1]==Wb):
            Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
            AnnMap = cv2.resize(AnnMap, dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,AnnMap
        # misc.imshow(Img)
######################################################Augmented Image for training##################################################################################################################################
    def Augment(self,Img,AnnMask,prob):
        Img=Img.astype(np.float)
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            AnnMask = np.fliplr(AnnMask)
        if np.random.rand()<0.5:
            Img = Img[..., :: -1]
        if np.random.rand()< 0.08: # flip up down
            Img=np.flipud(Img)
            AnnMask = np.flipud(AnnMask)
        if np.random.rand()< 0.08: # Rotate
            Img=np.rot90(Img)
            AnnMask = np.rot90(AnnMask)




        if np.random.rand() < 0.03: # resize
            r=r2=(0.3 + np.random.rand() * 1.7)
            if np.random.rand() < 0.1:
                r2=(0.5 + np.random.rand())
            h = int(Img.shape[0] * r)
            w = int(Img.shape[1] * r2)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

            AnnMask =  cv2.resize(AnnMask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        # if np.random.rand() < prob/3: # Add noise
        #     noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9
        #     Img *=noise
        #     Img[Img>255]=255
        #
        # if np.random.rand() < prob/3: # Gaussian blur
        #     Img = cv2.GaussianBlur(Img, (5, 5), 0)

        if np.random.rand() < 0.07:  # Dark light
            Img = Img * (0.5 + np.random.rand() * 0.65)
            Img[Img>255]=255

        if np.random.rand() < 0.10:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,AnnMask
########################################################################################################################################################
# ==========================Read single image annotation and data===============================================================================================
    def LoadNext(self, pos, Hb=-1, Wb=-1):
            Nim = np.random.randint(len(self.AnnList))
            AnnName = self.AnnList[Nim]
     #       CatSize = len(self.AnnList)
            AnnMask = cv2.imread(self.AnnDir+"/"+AnnName)
            Img = cv2.imread(self.ImageDir+"/"+AnnName.replace(".png",".jpg"))  # Load Image
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
#-------------------------Augment-----------------------------------------------------------------------------------------------
            Img,AnnMask=self.Augment(Img,AnnMask,0.1)
#-----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            if not Hb==-1:
               Img, AnnMask = self.CropResize(Img, AnnMask, Hb, Wb)
#-------------------------Read annotation--------------------------------------------------------------------------------
            self.BIgnore[pos]=(AnnMask[:,:,1]==1) # Second channel is ignore/ROI
            self.BImg[pos]=Img
            self.BAnnMaps[pos] = (AnnMask[:,:,0]==1) # First channel is binary vessel region mask

############################################################################################################################################################
# Start load batch of images (multi  thread the reading will occur in background and will will be ready once waitLoadBatch as be run)
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))


        
        #====================Start reading data multithreaded===========================================================
        self.BAnnMaps={}
        self.thread_list = []
        self.BIgnore = np.zeros([BatchSize, Hb, Wb], dtype=float)
        self.BImg = np.zeros([BatchSize, Hb, Wb,3], dtype=float)
        self.BAnnMaps= np.zeros([BatchSize, Hb, Wb], dtype=float)

        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="thread"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
        self.itr+=BatchSize
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
# Load batch for training (muti threaded  run in parallel with the training proccess)
# return previously  loaded batch and start loading new batch
    def LoadBatch(self):
            self.WaitLoadBatch()
            Imgs=self.BImg
            Ignore=self.BIgnore
            AnnMaps = self.BAnnMaps
            self.StartLoadBatch()
            return Imgs, Ignore, AnnMaps

