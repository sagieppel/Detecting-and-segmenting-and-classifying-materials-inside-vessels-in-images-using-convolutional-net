# Run instance and semantic segmentation on video  and save results to output video
###############################33
def show(Im):
    cv2.imshow('Main', Im.astype(np.uint8))
    cv2.waitKey(5000)
###############################################################################################################################
#-------------------------------------Input output folder-----------------------------------------------------------------------
import os
InputDir="//media/seppel/KINGSTON/tocpic/"#/scratch/gobi2/seppel/Chemscape/ChemLabScapeDataset/Simple/Test/Image/" # Folder with input images
#MainOutDir="/media/seppel/KINGSTON/OUT/"

MainOutDir="/scratch/gobi2/seppel/Chemscape/TocOutDir3/"
if not os.path.exists(MainOutDir): os.mkdir(MainOutDir)
#--------------------------------------Running pramters-------------------------------------------------------------------------------
UseGPU=True # run on GPU (true) or CPU (false) # Note this system is slow on GPU and very very slow on CPU
FreezeBatchNorm_EvalON=True # Freeze the upddating of bath normalization statitics -->  use net.eval()

VesIOUthresh=0.4#7 # IOU quality threshold for predicted vessel instance to be accepted
MatIOUthresh=0.1#5# IOU quality threshold for predicted material instance to be accepted
NumVessCycles=5 # Number of attempts to search for vessel instance, increase the probability to find vessel but also running time
NumMatCycles=6  # Number of attempts to search for material instance, increase the probability to find material phase but also running time
UseIsVessel=False#True # Only If the vessel instance net was trained with COCO  it can predict whether the instance belongs to a vessel  which can help to remove a false segment
IsVesThresh=0.5

#...........................................Trained net Paths.......................................................................
SemanticNetTrainedModelPath="Semantic/logs/1000000_Semantic_withCOCO_AllSets.torch"
InstanceVesselNetTrainedModelPath="InstanceVesselWithCOCO/logs/Vessel_Coco_610000_Trained_on_All_Sets.torch"
InstanceMaterialNetTrainedModelPath="InstanceMaterial/logs//Material_CocO_AllSets_1040000.torch"
#...............................Imports..................................................................

import os
import torch
import numpy as np
import Semantic.FCN_NetModel as SemanticNet
import Semantic.CategoryDictionary as CatDic
import InstanceVessel.FCN_NetModel as VesselInstNet
import InstanceMaterial.FCN_NetModel as MatInstNet

import cv2
import scipy.misc as misc


####################### List of classes#######################################################################
CatName={}
CatName[1]='Vessel'
CatName[2]='V Label'
CatName[3]='V Cork'
CatName[4]='V Parts GENERAL'
CatName[5]='Ignore'
CatName[6]='Liquid GENERAL'
CatName[7]='Liquid Suspension'
CatName[8]='Foam'
CatName[9]='Gel'
CatName[10]='Solid GENERAL'
CatName[11]='Granular'
CatName[12]='Powder'
CatName[13]='Solid Bulk'
CatName[14]='Vapor'
CatName[15]='Other Material'
CatName[16]='Filled'
MaterialCats={'Liquid GENERAL','Liquid Suspension','Foam','Gel','Solid GENERAL','Granular','Powder','Solid Bulk','Vapor','Other Material'}
PartsCats={'V Label','V Cork','V Parts GENERAL'}
###############################################################################################

#---------------------------open  video files-------------------------------------------------

#=========================Load Semantic net====================================================================================================================
print("Load semantic net")
SemNet=SemanticNet.Net(CatDic.CatNum) # Create net and load pretrained encoder path
if UseGPU:
     SemNet.load_state_dict(torch.load(SemanticNetTrainedModelPath))#180000.torch"))
else:
     SemNet.load_state_dict(torch.load(SemanticNetTrainedModelPath, map_location=torch.device('cpu')))  # 180000.torch"))

# SemNet.cuda()
# SemNet.half()
# #SemNet.eval()

# #=======================Load vessel Instance net======================================================================================================================
print("Load vessel instance  net")
VesNet=VesselInstNet.Net(NumClasses=2) # Create net and load pretrained
VesNet.AddEvaluationClassificationLayers(NumClass=1)
if UseGPU:
      VesNet.load_state_dict(torch.load(InstanceVesselNetTrainedModelPath))
else:
      VesNet.load_state_dict(torch.load(InstanceVesselNetTrainedModelPath, map_location=torch.device('cpu')))

# VesNet.cuda()
# VesNet.half()
# #VesNet.eval()

#====================Load Material instance net======================================================================================================================
#====================Load Material instance net======================================================================================================================
#import InstanceMaterialLoss_CatLossX2_ClassBlance03.FCN_NetModel as MatInstNet
print("Load material phase instance  net")
MatNet=MatInstNet.Net(NumClasses=2) # Create net and load pretrained
MatNet.AddEvaluationClassificationLayers(NumClass=20)
if UseGPU:
          MatNet.load_state_dict(torch.load(InstanceMaterialNetTrainedModelPath))
else:
    MatNet.load_state_dict(torch.load(InstanceMaterialNetTrainedModelPath, map_location=torch.device('cpu')))
# MatNet.cuda()
# MatNet.half()
# #MatNet.eval()

print("Finished loading nets")

############################################################################################################################################################################################################################################################
############################################################Split vessel region to vessel instances#########################################################################################################################################################
def FindVesselInstances(Img,VesselsMask): # Split the VesselMask into vessel instances using GES net for instances

            H,W=VesselsMask.shape
            InsList = np.zeros([0, H,W]) # list of vessels instances
            InstRank = [] # Score predicted for the instace
            InstMap = np.zeros([H,W],int) # map of instances that were already discovered
            NInst=0 # Number of instances
            OccupyMask=np.zeros([H,W],int) # Region that have been segmented
            ROIMask=VesselsMask.copy() # Region to be segmented
            NumPoints= int(340000 * 10/(H*W)) # Num instace points to guess per experiment
#===============Generate instance map========================================================================================
            for cycle in range(NumVessCycles):
                # ........................Generate input for the instance net,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                PointerMask=np.zeros([NumPoints,H,W],dtype=float) # Pointer mask
                ROI = np.ones([NumPoints, H, W], dtype=float)
                ImgList= np.ones([NumPoints, H, W,3], dtype=float)

                for i in range(NumPoints): # Generate pointer mask
                        while(True):
                            px = np.random.randint(W)
                            py = np.random.randint(H)
                            if (VesselsMask[py, px]) == 1: break
                        PointerMask[i,py,px]=1
                        ImgList[i]=Img
                        #ROI[i]=VesselsMask
                        # --------------------------------------
                # for f in range(1):#NumPoints):
                #     ImgList[f, :, :, 1] *= VesselsMask.astype(np.uint8)
                #     misc.imshow(Imgs[f])
                #     misc.imshow((ROI[f] + ROI[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
    #=====================================Run Net predict instance region their IOU score and wether they correspond to vessels or other objects============================================================================================================================
            ####    VesNet.train()#*******************************************
                with torch.autograd.no_grad():
                    Prob, Lb, PredIOU, PredIsVessel = VesNet.forward(Images=ImgList, Pointer=PointerMask,ROI=ROI,TrainMode=False,UseGPU=UseGPU, FreezeBatchNorm_EvalON=FreezeBatchNorm_EvalON)
    #======================================s========================================================================================================
                Masks = Lb.data.cpu().numpy().astype(float)
                IOU = PredIOU.data.cpu().numpy().astype(float)
                IsVessel = PredIsVessel.data.cpu().numpy().astype(float)[:,1]
     ###############################################################################################################################33
                # if IsVessel.min()<00.5:
                #     for i in range(IsVessel.shape[0]):
                #         if IsVessel[i]>0.5: continue
                #         print("IsVessel="+str(IsVessel[i])+"  IOU="+str(IOU[i][0]))
                #         Im = Img.copy()
                #         Im[:, :, 0] *= 1 - Masks[i].astype(np.uint8)
                #         Im[:, :, 1] *= 1 - Masks[i].astype(np.uint8)
                #         misc.imshow(Im)
                #     print("ggggggggggggg")

    ##################################Filter overlapping and low score segment############################################################################
                Accept=np.ones([NumPoints])
                for f in range(NumPoints):
                    SumMask=Masks[f].sum()
                    if IOU[f]<VesIOUthresh-cycle*0.05 or ((Masks[f]*OccupyMask).sum()/SumMask)>0.08:
                           Accept[f]=0
                           continue
                    for i in range(NumPoints):
                        if i==f: continue
                        if IOU[f] > IOU[i] or Accept[i]==0: continue
                        fr=(Masks[i]*Masks[f]).sum()/SumMask
                        if  (fr>0.05):
                                    Accept[f]=0
                                    break

    #===================================================Remove  predictions that over lap previous prediction========================================================================================================================
                for f in range(NumPoints):
                    if Accept[f]==0: continue
                    OverLap = Masks[f] * OccupyMask
                    if (OverLap.sum() > 0):
                        Masks[f][OverLap>0] = 0
                    for i in range(NumPoints):
                        if Accept[i] == 0 or i==f or  IOU[f]>IOU[i]: continue
                        OverLap=Masks[i]*Masks[f]
                        fr=(OverLap).sum()
                        if  (fr>0):  Masks[f][OverLap>0]=0
    #=============================Add selected mask to final segmentatiomn map and instance list=======================================================================================================================================
                for f in range(NumPoints):
                        if Accept[f]==0: continue
                        if (IsVessel[f] > IsVesThresh or not UseIsVessel):
                            NInst+=1
                            InsList = np.concatenate([InsList,np.expand_dims(Masks[f],0)],axis=0)
                            InstRank.append(IOU[f])
                            InstMap[Masks[f]>0]=NInst
                        OccupyMask[Masks[f]>0]=1
    #=============================================================================================================================================================================================
                print("cycle"+str(cycle))
                # for i in range(NInst):
                #     print(InstRank[i])
                #     Img2=Img.copy()
                #     Img2[:, :, 1] *= 1 - Masks[i].astype(np.uint8)
                #     Img2[:, :, 0] *= 1 - Masks[i].astype(np.uint8)
                #
                #     misc.imshow(cv2.resize(np.concatenate([Img,Img2],axis=1),(1000,500)))


    #===============================================Update ROI mask==============================================================================================================================================
                ROIMask[OccupyMask>0]=0
                if (ROIMask.sum()/ VesselsMask.sum())<0.03: break

            # Img2 = Img.copy()
            # for i in range(InstMap.max()):
            #  Img2[:, :, 0][InstMap==i]+=i*30
            # Img2[:, :, 1] = 0.5 * Img2[:, :, 1] + 0.5 * (InstMap * 50).astype(np.uint8)
            # Img2[:, :, 2] = 0.5 * Img2[:, :, 2] + 0.5 * (InstMap * 93).astype(np.uint8)
            # print(NInst)
            # misc.imshow(InstMap * 30)
            # misc.imshow(cv2.resize(np.concatenate([Img, Img2], axis=1), (1000, 500)))

            return InsList,InstRank,InstMap, OccupyMask,NInst
###########################################################################################################################################################################################################################################
#====================================================================================================================================================================
############################################################Split  vessel region to materials instances#########################################################################################################################################################
def FindMaterialInstances(Img,VesselsMask): # Split  vessel region to materials instance and empyty region instances and classify material

            H,W=VesselsMask.shape
            InsList = np.zeros([0, H,W]) #List of instances
            InsCatList = np.zeros([0,20]) # Class list for instances
            InstRank = [] # IOU score for the instances
            InstMap = np.zeros([H,W],int)
            NInst=0
            OccupyMask=np.zeros([H,W],int) # Region that already been segmented
            ROIMask=VesselsMask.copy() # Region to be segmented
            NumPoints= int(340000 * 10/(H*W)) # Num instace points to guess per experiment
#===============Generate instance map========================================================================================
            for cycle in range(NumMatCycles):
# .........................Generate input for the net................................................................
                PointerMask=np.zeros([NumPoints,H,W],dtype=float)
                ROI = np.ones([NumPoints, H, W], dtype=float)
                ImgList= np.ones([NumPoints, H, W,3], dtype=float)
                for i in range(NumPoints): # generate pointer mask
                        while(True):
                            px = np.random.randint(W)
                            py = np.random.randint(H)
                            if (VesselsMask[py, px]) == 1: break
                        PointerMask[i,py,px]=1
                        ImgList[i]=Img
                        ROI[i]=VesselsMask
                        # --------------------------------------
                # for f in range(1):#NumPoints):
                #     ImgList[f, :, :, 1] *= VesselsMask.astype(np.uint8)
                #     misc.imshow(Imgs[f])
                #     misc.imshow((ROI[f] + ROI[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
    #=====================================Run Net============================================================================================================================
                with torch.autograd.no_grad():
                    Prob, Lb, PredIOU, Predclasslist = MatNet.forward(Images=ImgList, Pointer=PointerMask,ROI=ROI,TrainMode=False,UseGPU=UseGPU, FreezeBatchNorm_EvalON=FreezeBatchNorm_EvalON)

    #======================================Filter overlapping and low score predictions========================================================================================================
                Masks = Lb.data.cpu().numpy().astype(float)
                IOU = PredIOU.data.cpu().numpy().astype(float)

                ClassVector=np.zeros([Predclasslist[0].shape[0],20])
                for ic in range(len(Predclasslist)):
                    ClassPr = Predclasslist[ic].data.cpu().numpy()[:,1]>0.5
                    ClassVector[:,ic]=ClassPr

                Accept=np.ones([NumPoints])
                for f in range(NumPoints):
                    SumMask=Masks[f].sum()
                    if IOU[f]<MatIOUthresh-cycle*0.1 or ((Masks[f]*OccupyMask).sum()/SumMask)>0.08:
                           Accept[f]=0
                           continue
                    for i in range(NumPoints):
                        if i==f: continue
                        if IOU[f] > IOU[i] or Accept[i]==0: continue
                        fr=(Masks[i]*Masks[f]).sum()/(SumMask+0.00001)
                        if  (fr>0.05):
                                    Accept[f]=0
                                    break

    #===================================================Remove instace that overlap previously annotated region========================================================================================================================
                for f in range(NumPoints):
                    if Accept[f]==0: continue
                    OverLap = Masks[f] * OccupyMask
                    if (OverLap.sum() > 0):
                        Masks[f][OverLap>0] = 0
                    for i in range(NumPoints):
                        if Accept[i] == 0 or i==f or  IOU[f]>IOU[i]: continue
                        OverLap=Masks[i]*Masks[f]
                        fr=(OverLap).sum()
                        if  (fr>0):  Masks[f][OverLap>0]=0


    #=============================Add selected mask to final segmentatiomn=======================================================================================================================================
                for f in range(NumPoints):
                        if Accept[f]==0: continue
                        NInst+=1
                      ##  InstCat.append(ClassPr[f])
                        InsList = np.concatenate([InsList,np.expand_dims(Masks[f],0)],axis=0)
                        InsCatList = np.concatenate([InsCatList, np.expand_dims(ClassVector[f], 0)], axis=0)
                        InstRank.append(IOU[f])
                        InstMap[Masks[f]>0]=NInst
                        OccupyMask[Masks[f]>0]=1
    #=============================================================================================================================================================================================
                print("cycle"+str(cycle))
                # for i in range(NInst):
                #     print(InstRank[i])
                #     Img2=Img.copy()
                #     Img2[:, :, 1] *= 1 - Masks[i].astype(np.uint8)
                #     Img2[:, :, 0] *= 1 - Masks[i].astype(np.uint8)
                #
                #     misc.imshow(cv2.resize(np.concatenate([Img,Img2],axis=1),(1000,500)))


    #===============================================Update ROI mask==============================================================================================================================================
                ROIMask[OccupyMask>0]=0
                if (ROIMask.sum()/ VesselsMask.sum())<0.03: break

            # Img2 = Img.copy()
            # for i in range(InstMap.max()):
            #  Img2[:, :, 0][InstMap==i]+=i*30
            # Img2[:, :, 1] = 0.5 * Img2[:, :, 1] + 0.5 * (InstMap * 50).astype(np.uint8)
            # Img2[:, :, 2] = 0.5 * Img2[:, :, 2] + 0.5 * (InstMap * 93).astype(np.uint8)
            # print(NInst)
            # misc.imshow(InstMap * 30)
            # misc.imshow(cv2.resize(np.concatenate([Img, Img2], axis=1), (1000, 500)))

            return InsList,InstRank,InstMap, OccupyMask,NInst,InsCatList


##############################################################################################################################################################################################################################################
      ##                                                               MAIN
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
      ##                                                               MAIN
##############################################################################################################################################################################################################################################
# OutDir1=MainOutDir+"/1/"
# OutDir2=MainOutDir+"/2/"
# OutDir3=MainOutDir+"/3/"
# OutDir4=MainOutDir+"/4/"
# OutDir5=MainOutDir+"/5/"
# OutDir6=MainOutDir+"/6/"

# if not os.path.isdir(OutDir1): os.mkdir(OutDir1)
# if not os.path.isdir(OutDir2): os.mkdir(OutDir2)
# if not os.path.isdir(OutDir3): os.mkdir(OutDir3)
# if not os.path.isdir(OutDir4): os.mkdir(OutDir4)
# if not os.path.isdir(OutDir5): os.mkdir(OutDir5)
# if not os.path.isdir(OutDir6): os.mkdir(OutDir6)






for FileName in os.listdir(InputDir):


        Im = cv2.imread(InputDir+"/"+FileName)
        #nframe += 1

        h0,w0,d=Im.shape
        r=np.max([h0,w0])
        print(Im.shape)
        if r>840:
            fr=840/r
            Im=cv2.resize(Im,(int(w0*fr),int(h0*fr)))
        # if r<200:
        #     fr=200/r
        #     Im=cv2.resize(Im,(int(w0*fr),int(h0*fr)))
        Imgs=np.expand_dims(Im,axis=0)

        if not (type(Im) is np.ndarray): continue
    #################################################################################################################################################################
    #=====================================================================Semantic============================================================================================================================================================================
    # =====================================================================Semantic============================================================================================================================================================================
    # =====================================================================Semantic============================================================================================================================================================================
        print("Applying semantic segmentation")
        with torch.autograd.no_grad():
              OutProbDict,OutLbDict=SemNet.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNorm_EvalON) # Run semntic net inference and get prediction

    #==============Create and save annotation map for each class====================================================================================================

        CatMap={}
        for nm in OutLbDict:
            Lb=OutLbDict[nm].data.cpu().numpy()[0].astype(np.uint8)
            if Lb.mean()<0.01: continue
            if nm=='Ignore': continue
            CatMap[nm]=Lb
            if  nm=='Filled': break


    #####################################################Instance segmentation#####################################################################################################
    #####################################################Instance segmentation#####################################################################################################
     #------------------------Find Vessel instance take the vessel region find in the semantic segmentation and split it into individual vessel instances--------------------------------------------------------------------------------------------------------------------
        print("Applying Vessel instance segmentation")
        NumVess=0
        NumMat=0
        NumPart=0
        OutAnnMap = np.zeros([Imgs[0].shape[0], Imgs[0].shape[1], 3], dtype=np.uint8)

        VesselRegion=OutLbDict['Vessel'].data.cpu().numpy()[0].astype(float)
        if VesselRegion.mean()<0.001:
                   VesInsList=[]
                   InstRank=[]
                   InstMapVes=[]
                   OccupyMask=np.zeros([Imgs[0].shape[0], Imgs[0].shape[1]])
                   NInst=0
        else:
                   VesInsList,InstRank,InstMapVes, OccupyMask,NInst=FindVesselInstances(Imgs[0],VesselRegion)




    ########################################Material instance segmentation, take the region of each vessel instance and split it into different material phases and empty regions#################################################################################################################
        print("Applying material instance segmentation")

        CatMap2={} # Semantic Map that will be generated by uniting the cats of different instance
        for ff,VesIns in enumerate(VesInsList): # go over all vessel instances and find the material instances inside this vessels
           NumVess+=1

           OutAnnMap[:,:,2][VesIns>0]=NumVess
           MatInsList,InstRank,InstMap, OccupyMask,NInst,InsCatList=FindMaterialInstances(Imgs[0],VesIns)

    # ***********************************************************************************************************************************************************************************************************
    #--------------------------------add material class to json dictionary and create material instance map------------------------------------------------------------------------------------------------------------------------------
           for ff,MatIns in enumerate(MatInsList):
                  print("nmat")
                  print(NumMat)
                  if InsCatList[ff][1:].sum()>0 and InsCatList[ff][0]==0:
                         NumMat += 1
                         OutAnnMap[:, :, 0][MatIns > 0] = NumMat
                         # CatDic[AnnName]["MaterialCats"][NumMat] = []
                         # for ic in  range(len(InsCatList[ff])):
                         #     if (InsCatList[ff,ic]>0):
                         #          CatDic[AnnName]["MaterialCats"][NumMat].append(CatName[ic])



    #--------------------------------------Find Part Cats (basically take the parts region from the semantic segmentation and split them using connected component (and assume a connected component is instance)------------------------------------------------------------------------------------------------------------
           for nm in CatMap:
                if not (nm in PartsCats): continue
                PIns=VesIns * CatMap[nm]
                fr = PIns.sum() / VesIns.sum()

                if (fr>0.01) and PIns.sum()>81:
                            ex=False
                            for kk in range(1,OutAnnMap[:, :, 1].max()+1): # Check if exists
                                if ((PIns*(OutAnnMap[:, :, 1]==kk)).sum()/PIns.sum())>0.5:
                                      ex=True
                                  #    CatDic[AnnName]["PartCats"][kk].append(nm)

                            if not ex:
                                NumPart+=1
                                OutAnnMap[:, :, 1][(VesIns * CatMap[nm])>0] = NumPart
                                # CatDic[AnnName]["PartCats"][NumPart]=[]
                                # CatDic[AnnName]["PartCats"][NumPart].append(nm)
    #------------------------------------Save instance map--------------------------------------------------------------------------------------------------------
    # --------------------Convert new annotation map to match old annotation map for frame consistancy basically phase Tracker--------------------------------------------------------------------------------------
        TRate=0.2

        #-------------------------------------Save second type semantic maps--------------------------------------------------------------------------------------
        #---------------Save instance annotation  overlay on image for vizuallization---------------------------------------------------------------------------------------------------------
        for Tt in range(0,6):
            TRate = 0.1*Tt
            h,w,d=Im.shape
            InsVizVessl = Im.copy()
            InsVizVessl[:, :, 0] = np.uint8((OutAnnMap[:, :, 2] * 21) % 255)
            InsVizVessl[:, :, 1] = np.uint8((OutAnnMap[:, :, 2] * 667) % 255)
            InsVizVessl[:, :, 2] = np.uint8((OutAnnMap[:, :, 2] * 111) % 255)
            InsVizVessl = InsVizVessl * (1-TRate) + Im * TRate

            InsVizMat = Im.copy()
            InsVizMat[:, :, 0] = np.uint8((OutAnnMap[:, :, 0] * 21) % 255)
            InsVizMat[:, :, 1] = np.uint8((OutAnnMap[:, :, 0] * 667) % 255)
            InsVizMat[:, :, 2] = np.uint8((OutAnnMap[:, :, 0] * 111) % 255)
            InsVizMat = InsVizMat * (1-TRate) + Im * TRate

            InsVizVesPart = Im.copy()
            InsVizVesPart[:, :, 0] = np.uint8((OutAnnMap[:, :, 1] * 21) % 255)
            InsVizVesPart[:, :, 1] = np.uint8((OutAnnMap[:, :, 1] * 667) % 255)
            InsVizVesPart[:, :, 2] = np.uint8((OutAnnMap[:, :, 1] * 111) % 255)
            InsVizVesPart = InsVizVesPart * (1-TRate) + Im * TRate
            # cv2.putText(InsVizVessl, "Vessels", (int(w / 3), int(h / 6)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(InsVizVesPart, "Part", (int(w / 3), int(h / 6)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2,cv2.LINE_AA)
            # cv2.putText(InsVizMat, "Materials", (int(w / 3), int(h / 6)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2,cv2.LINE_AA)

            VizImg = np.concatenate([Im,np.ones([Im.shape[0],30,3])*255,InsVizVessl], axis=1).astype(np.uint8)
            VizImg2 = np.concatenate([InsVizMat,np.ones([Im.shape[0],30,3])*255, InsVizVesPart], axis=1).astype(np.uint8)
            VizImg = np.concatenate([VizImg,np.ones([30,VizImg.shape[1],3])*255, VizImg2], axis=0).astype(np.uint8)
            OutFolder=MainOutDir+"/Overlay"+str(TRate)+"/"
            if not os.path.isdir(OutFolder): os.mkdir(OutFolder)
            cv2.imwrite(OutFolder+"/"+FileName.replace(".","_")+".png",VizImg)

        #------------------------------------------------------------------------------------------------------------------------
        for it in range(4,14,2):

            h,w,d=Im.shape
         #   InsVizGen = Im.copy()
            InsVizVessl = Im.copy()
            for yy in range(1,OutAnnMap[:, :, 2].max()+1):
                Msk=(OutAnnMap[:, :, 2]==yy).astype(np.uint8)
#                cv2.fi
#                Msk = cv2.imfill(Msk, 'holes')
                kernel = np.ones((3, 3), np.uint8)
                Msk -= cv2.erode(Msk, kernel, iterations=it)

                InsVizVessl[:, :, 0][Msk>0] = np.uint8((yy* 21) % 255)
                InsVizVessl[:, :, 1][Msk>0] = np.uint8((yy * 667) % 255)
                InsVizVessl[:, :, 2][Msk>0] = np.uint8((yy * 111) % 255)

            InsVizGen= InsVizVessl.copy()

            InsVizMat = Im.copy()
            for yy in range(1, OutAnnMap[:, :, 0].max() + 1):
                Msk = (OutAnnMap[:, :, 0] == yy).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                Msk1 =  cv2.erode(Msk, kernel, iterations=int(it/2))-cv2.erode(Msk, kernel, iterations=it)
                Msk =Msk- cv2.erode(Msk, kernel, iterations=it)

                InsVizMat[:, :, 0][Msk > 0] = np.uint8((yy * 173) % 255)
                InsVizMat[:, :, 1][Msk > 0] = np.uint8((yy * 28) % 255)
                InsVizMat[:, :, 2][Msk > 0] = np.uint8((yy * 967) % 255)

                InsVizGen[:, :, 0][Msk1 > 0] = np.uint8((yy * 173) % 255)
                InsVizGen[:, :, 1][Msk1 > 0] = np.uint8((yy * 28) % 255)
                InsVizGen[:, :, 2][Msk1 > 0] = np.uint8((yy * 967) % 255)




            VizImg = np.concatenate([Im,np.ones([Im.shape[0],30,3])*255,InsVizVessl], axis=1).astype(np.uint8)
            VizImg2 = np.concatenate([InsVizMat,np.ones([Im.shape[0],30,3])*255, InsVizGen], axis=1).astype(np.uint8)
            VizImg = np.concatenate([VizImg,np.ones([30,VizImg.shape[1],3])*255, VizImg2], axis=0).astype(np.uint8)
            OutFolder=MainOutDir+"/Contour_DilRate"+str(it)+"/"
            if not os.path.isdir(OutFolder): os.mkdir(OutFolder)
            cv2.imwrite(OutFolder+"/"+FileName.replace(".","_")+".png",VizImg)





