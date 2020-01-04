import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
import random
import ChemReader as ChemReader
import torch
######################Class used for evaluating net accuracy######################################################################################
#########################################################################################################################
class Evaluator:
    def __init__(self, AnnDir,OutFile):
        self.AnnDir = AnnDir
        self.OutFile=OutFile
        f=open(OutFile,"w")
        f.close()
        print("-------------------------------------Creating test evaluator------------------------------------------------------")
        self.Reader = ChemReader.Reader(MainDir=self.AnnDir, TrainingMode=False)
##########################################################Evaluating net accuracy###########################333
    def Eval(self,Net,itr):
        print("Evaluating")
        Finished=False
        IOUSum={}
        InterSum={}
        UnionSum={}
        ImSum={}
        while (not Finished):
                Img,AnnMap,Ignore,Finished=self.Reader.LoadSingle()
                Img=np.expand_dims(Img,axis=0)
                ROI = 1 - Ignore
                with torch.autograd.no_grad():
                         OutProbDict, OutLbDict = Net.forward(Images=Img, TrainMode=True)
                         if not IOUSum:
                             for nm in AnnMap:
                                 IOUSum[nm]=0
                                 InterSum[nm]=0
                                 UnionSum[nm]=0
                                 ImSum[nm]=0
                         for nm in AnnMap:
                             Pred=OutLbDict[nm].data.cpu().numpy()[0]*ROI
                             GT=AnnMap[nm][:,:,0]*ROI
                             Inter=(Pred*GT).sum()
                             Union=(Pred).sum()+(GT).sum()-Inter
                             if Union>0:
                                IOUSum[nm] += Inter/Union
                                InterSum[nm] += Inter
                                UnionSum[nm] += Union
                                ImSum[nm] += 1


                                # if GT.sum()>0:
                                #     print(nm)
                                #     Im=Img[0].copy()
                                #     Im[:, :, 0] *= 1 - GT.astype(np.uint8)
                                #     Im[:, :, 2] *= ROI.astype(np.uint8)
                                #     Im[:, :, 1] *= 1 - Pred.astype(np.uint8)
                                #     misc.imshow(Im)


        f = open(self.OutFile, "a")
        txt="\n=================================================================================\n"
        txt+=str(itr)+"\n"
        for nm in IOUSum:
            if UnionSum[nm]>0:
                txt += nm + "\t"
                txt += "IOU Average Per Pixel=\t"+str(InterSum[nm]/UnionSum[nm])+"\t"
                txt += "IOU Average Per Image=\t" + str(IOUSum[nm]/ImSum[nm])+"\n"
        f.write(txt)
        f.close()
        print(txt)






