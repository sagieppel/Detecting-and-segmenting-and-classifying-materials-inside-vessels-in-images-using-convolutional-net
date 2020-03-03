#Run  trained net on video to generate prediction and write to another video
#...............................Imports..................................................................

import os
import torch
import numpy as np
import FCN_NetModel as FCN # The net Class
import CategoryDictionary as CatDic
import cv2
#import scipy.misc as misc

############################################Input parameters###################################################################################
#-------------------------------------Input parameters-----------------------------------------------------------------------
InputFolder=r"/media/sagi/DefectiveHD/Medical/Blood//" #input video
OutDir=r"/media/sagi/DefectiveHD/Medical/BloodOUT//"
OutVideoAll=r"/media/sagi/DefectiveHD/Medical/BloodAll.avi"
OutVideoMAIN=r"/media/sagi/DefectiveHD/Medical/Blood.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
AllCatsVideoWriter = cv2.VideoWriter(OutVideoAll, fourcc, 0.6, (1404, 1404))
MainCatsVideoWriter = cv2.VideoWriter(OutVideoMAIN, fourcc, 0.6, (1404, 704))
if not os.path.exists(OutDir): os.makedirs(OutDir) # Create folder for trained weight

UseGPU=True # Use GPU or CPU  for prediction (GPU faster but demend nvidia GPU and CUDA installed else set UseGPU to False)
FreezeBatchNormStatistics=False # wether to freeze the batch statics on prediction  setting this true or false might change the prediction mostly False work better
OutEnding="" # Add This to file name

#-----------------------------------------Location of the pretrain model-----------------------------------------------------------------------------------
Trained_model_path =r"logs//TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch"

##################################Load net###########################################################################################
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=FCN.Net(CatDic.CatNum) # Create net and load pretrained encoder path
if UseGPU==True:
    print("USING GPU")
    Net.load_state_dict(torch.load(Trained_model_path))
else:
    print("USING CPU")
    Net.load_state_dict(torch.load(Trained_model_path, map_location=torch.device('cpu')))
#---------------------OPEN video-----------------------------------------------------------------------------------------------------

#--------------------Create output video---------------------------------------------------------------------------------


#-----------------------Read Frame one by one-----------------------------------------------------------------------
# Read until video is completed
#iii=0
for name in os.listdir(InputFolder):
   # if iii>3: break
    # Capture frame-by-frame
    # ..................Read and resize image...............................................................................

    Im = cv2.imread(InputFolder+"/"+name)

    if Im is None: continue
        # Display the resulting frame

    h,w,d=Im.shape
    r=np.max([h,w])
    if r>840: # Image larger then 840X840 are shrinked (this is not essential, but the net results might degrade when using to large images
        fr=840/r
        Im=cv2.resize(Im,(int(w*fr),int(h*fr)))
    h, w, d = Im.shape
    Imgs=np.expand_dims(Im,axis=0)
    if not (type(Im) is np.ndarray): continue
#................................Make Prediction.............................................................................................................
    with torch.autograd.no_grad():
          OutProbDict,OutLbDict=Net.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNormStatistics) # Run net inference and get prediction
#------------------------------------Display main classes on the image----------------------------------------------------------------------------------

    my=1
    mx=3
    OutMain = np.zeros([h * my, w * mx, 3], np.uint8)
    y = 0
    x = 1
    OutMain[:h,:w]=Im
    MainCatName = ['Vessel','Filled']
    VesMat = OutLbDict['Vessel'].data.cpu().numpy()[0].astype(np.uint8)
    for nm in MainCatName:
        Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
        #if Lb.mean()<0.001: continue
        if nm=='Ignore': continue
        font = cv2.FONT_HERSHEY_SIMPLEX
        nm=nm.replace("V ", "").replace("Liquid Suspension", "Suspension").replace(" GENERAL", "")
        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 1][Lb==1] = 0
        cv2.putText(ImOverlay1, nm, ( int(w/3), int(h/6)), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        ImOverlay1[:, :, 0][Lb==1] = 255
        OutMain[h*y:h*(y+1), w*x:w*(x+1)] = ImOverlay1
        x+=1
        if x>=mx:
            x=0
            y+=1
    h,w,d=OutMain.shape
    cv2.imwrite(OutDir + "/" + name[:-4] + "_Main.png", OutMain)
    fr = np.max([h/700, w/1400])
    OIm=np.zeros([704,1404,3],np.uint8)
    OutMain=cv2.resize(OutMain,(int(w/fr),int(h/fr)))
    h, w, d = OutMain.shape
    OIm[0:h,0:w,:]=OutMain
    print(OIm.shape)
    MainCatsVideoWriter.write(OIm)
    cv2.imwrite(OutDir + "/" + name[:-4] + "_Main.png", OIm)

    cv2.imshow('Main Classes', OIm)

    cv2.waitKey(25)


#------------------------------------Display all classes on the image----------------------------------------------------------------------------------
    h, w, d = Im.shape
    my=2
    mx=2
    OutMain = np.zeros([h * my, w * mx, 3], np.uint8)
    y = 0
    x = 1
    OutMain[:h,:w]=Im
    AllCatName = ['Vessel','Liquid GENERAL','V Label','V Cork','Liquid Suspension','Solid GENERAL']

    VesMat = OutLbDict['Vessel'].data.cpu().numpy()[0].astype(np.uint8)
    for nm in AllCatName:
        Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
        if Lb.mean()<0.0002: continue
        if nm=='Ignore': continue
        font = cv2.FONT_HERSHEY_SIMPLEX
        nm=nm.replace("V ","").replace("Liquid Suspension","Suspension").replace(" GENERAL","")
        ImOverlay1 = Im.copy()
        ImOverlay1[:, :, 1][Lb==1] = 0
        cv2.putText(ImOverlay1, nm, ( int(w/3), int(h/6)), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
        ImOverlay1[:, :, 0][Lb==1] = 255

        ImOverlay1[:, :, 2][Lb == 1] -= 20
        OutMain[h*y:h*(y+1), w*x:w*(x+1)] = ImOverlay1
        x+=1
        if x>=mx:
            x=0
            y+=1
        if y>=my: break
    h, w, d = OutMain.shape

    fr = np.max([h / 1400, w / 1400])
    OIm = np.zeros([1404, 1404, 3], np.uint8)
    OutMain = cv2.resize(OutMain, (int(w / fr), int(h / fr)))
    h, w, d = OutMain.shape
    OIm[0:h, 0:w, :] = OutMain
    AllCatsVideoWriter.write(OIm)
    cv2.imwrite(OutDir + "/" + name[:-4] + "_All.png", OIm)
    print(OIm.shape)
    cv2.imshow('ALl Classes', OIm)

    cv2.waitKey(25)
MainCatsVideoWriter.release()
AllCatsVideoWriter.release()
#
#
#
#
#
#
#
