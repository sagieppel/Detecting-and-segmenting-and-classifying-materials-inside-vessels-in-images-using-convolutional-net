#  Detecting, segmenting, and classifying materials inside vessels in images using a fully convolutional neural net, for chemistry laboratory and general setting.


Neural net that given an image, detects and segments and classifies the vessels (mainly transparent vessels) and the materials inside the vessels in the image (Figure 1). The net marks the vessel region, the filled region inside the vessel, and the specific region of various a phase of materials such as liquid, solid, foam, suspension, powder, granular... In addition, the net also predicts the region of the vessel labels cork and other parts (such as valves in separatory funnels). Note this is a semantic segmentation net based on PSP net.

See paper [Computer vision for recognition of materials andvessels in chemistry lab settings and theVector-LabPics dataset](https://chemrxiv.org/articles/Computer_Vision_for_Recognition_of_Materials_and_Vessels_in_Chemistry_Lab_Settings_and_the_Vector-LabPics_Dataset/11930004) for more details on the method and dataset.

### This net with a pretrained model that can be run out of the box, without training can be download from [here](https://zenodo.org/record/3697767) or [here](https://drive.google.com/file/d/1wWGPoa7aKBlvml6Awe4AzJUbNlR72K6X/view?usp=sharing).

# General
The net focus on detecting vessels and their content materials in images. The focus is on both chemistry lab setting and general everyday setting (beverage, kitchen..) but should work in any conditions or setting. The net should recognize any transparent vessel (bottle/glass /or lab vessel) and their content and some none transparent vessels in any general environment and setting. The accuracy of the net is relatively high in detecting and classifying vessels, filled regions, liquid regions, and solid regions. The classification accuracy for fine-grained material classes such as foams, powder, gels, etc., is lower. If you encounter cases on which the net performs badly, please send me the images so I can use them to improve the network.


 ![](/Figure1.jpg)
Figure 1) Input images and output results of the net. Images taken from the [NileRed](https://www.youtube.com/user/TheRedNile) youtube channel.
 


# Input and output of the net
The input for the net is a standard image (Figure 1 right).
The net output of the region of the vessel/fill level and other materials phases, and vessel parts in the image (Figure 1 left). For each class, the net will output a mask mask region of the image corresponding to this class in the image (Figure 1 left).








# Requirements
## Hardware
For using the trained net, no specific hardware is needed, but the net will run much faster on Nvidia GPU.

For training the net an Nvidia GPU is needed (the net was trained on Titan XP, and also on RTX 2070 with similar results)

## Software:
This network was run with Python 3.7 [Anaconda](https://www.anaconda.com/download/) with  [Pytorch](https://pytorch.org/) and OpenCV packages.




# Setup for running prediction
1) Install [Anaconda](https://www.anaconda.com/download/)
2) Create a virtual environment with the required dependencies ([Pytorch](https://pytorch.org/), torchvision, scipy and OpenCV): *conda env create -f environment.yml*
3) Activate the virtual environment: *conda activate vessel-segmentation*
4) Download the code with trained model weight from [here](https://zenodo.org/record/3697767) or [here](https://drive.google.com/file/d/1wWGPoa7aKBlvml6Awe4AzJUbNlR72K6X/view?usp=sharing).


# Tutorial


# Running inference on image and predicting segment mask
1. Download the code with trained model weight from [here](https://zenodo.org/record/3697767) or [here](https://drive.google.com/file/d/1wWGPoa7aKBlvml6Awe4AzJUbNlR72K6X/view?usp=sharing). or train the model yourself using the instructions of the Training section.
2. Prepare a folder with the input images (they should be in .jpg or .png format).
3. Run the RunPredictionOnFolder.py script (all the arguments have default values that are automatically set if none specified from the command line):  
    *python RunPredictionOnFolder.py --inputdir <input_dir_path> --outdir <out_dir_path> --gpu <True or False> --freeze <True or False> --trainedmodel <trained_model_path_and_name>* 
4. The output is the predicted region for each input image and class: it would appear in the *outdir* folder.

Note: RunPredictionOnFolder.py should run out of the box (as is) using the sample images and [trained model](https://drive.google.com/file/d/1AtZFRyKAiEk9Pfip636_c7tZJjT0xUOP/view?usp=sharing) provided.
  ## Notes on some arguments:
- If you train the net yourself, set the path to your trained model in the *trainedmodel* argument.
- If you have a Nvidia GPU and Cuda installed, set the *gpu* argument to True (this will allow the net to achieve a much faster running time).
- Changing the *freeze* argument from False to True might change the segmentation quality for better or worst (and so does changing the image size).

## Additional Running scripts, running on videos and webcam:
* RunPredictionOnVideo.py script: receive an Input video in InputVideo apply prediction overlay the prediction on the image  and save it to video files.

* RunPredictionWebCam.py script: Take image from webcam run prediction overlay the prediction on the image  and display on screen

# Training general
There are two training options: one is to train using only with LabPics dataset, this is faster, simpler. The second training option is to use a combination of the LabPics dataset and Vessels classes from the [COCO panoptic dataset](http://cocodataset.org/#download) (Such as bottles/glasses/jars..). This option is more complex to train and gives lower accuracy on the test set but gives a more robust net that work under a wider set of conditions. 


# Training simple (only LabPics)
1. Download the LabPics data set from [Here](https://zenodo.org/record/3697452) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)
2. Open the Train.py script
3. Set the path to the LabPics dataset main folder to the TrainFolderPath parameter.
4. Run the script 
5. Output trained model will appear in the /log subfolder or any folder set in Trained model Path



## Training second option (With LabPics dataset and Vessels from  the COCO panoptic dataset)
### Downloading datasets
1. Download the LabPics data set from [Here](https://zenodo.org/record/3697452) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)


2. Download the [COCO panoptic dataset](http://cocodataset.org/#download) annotation and train images.
### Converting COCO dataset into training data
3. Open script TrainingDataGenerationCOCO/RunDataGeneration.py
4. Set the COCO dataset image folder to the ImageDir parameter.
5. Set the COCO panoptic annotation folder to the AnnotationDir parameter.
6. Set the COCO panoptic .json file to the DataFile parameter.
7. Set the output folder (where the generated data will be saved) to the OutDir parameter.
8. Run script. 
### Training
9. Open the COCO_Train.py script
10. Set the path to the LabPics dataset main folder to the LabPicsTrainFolderPath parameters.
11. Set the path to the COCO generated data (OutDir, step 7)  to the COCO_TrainDir paramter.
12. Run the script 
13. Output trained model will appear in the /log_COCO subfolder or any folder set in Trained model Path





# Code file structure
RunPredictionOnFolder.py: Run prediction on image using pre-trained image

Train.py: Training the net of the [LabPics](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) dataset

ChemReader.py: File reader for the LabPics dataset (used by the Train.py script)

FCN_NetModel.py: The class containing the neural net model.

Evaluator.py: Evaluate the net performance during training (Used by Train.py)

CategoryDictionary.py: List of classes and subclasses used by the net and LabPics dataset.

Logs folder: Folder where the trained models and training logs are stored.

InputImages Folder: Example input images for the net.

### For second training mode (with COCO)

COCO_TRAIN.py:  Training script for second training mode (with COCO).

CocoReader.py: Reader for the converted COCO data.

TrainingDataGenerationCOCO folder: Convert COCO dataset for training data.
### Results on videos
Results on of the nets on videos can be seen here:
https://www.youtube.com/playlist?list=PLRiTwBVzSM3B6MirlFl6fW0YQR4TtQmtJ

# Links
LabPics dataset for annotated images of liquid, solid and foam materials in mostly transperent vessels in Lab setting and general everyday setting can be download from [here](https://drive.google.com/file/d/1TZao7JDzxcJr_hMqYHLRcV2N0UHoH2c1/view?usp=sharing) or [here](https://drive.google.com/file/d/1gfaM_6eZjtg7dkFShGl1gIfsXzj1KjIX/view?usp=sharing)

Train model for this net can be download from [here](https://zenodo.org/record/3697767) or [here](https://drive.google.com/file/d/1wWGPoa7aKBlvml6Awe4AzJUbNlR72K6X/view?usp=sharing)..

# Thanks

The images for the [LabPics dataset](https://zenodo.org/record/3697452) were supplied by the following sources Nessa Carson (@SuperScienceGrl Twitter), Chemical and Engineering Science chemistry in pictures, YouTube channels dedicated to chemistry experiments: NurdRage, NileRed, DougsLab, ChemPlayer, and Koen2All. Additional sources for images include Instagram channels chemistrylover_(Joana Kulizic),Chemistry.shz (Dr.Shakerizadeh-shirazi), MinistryOfChemistry, Chemistry And Me, ChemistryLifeStyle, vacuum_distillation, and Organic_Chemistry_Lab

Work was done in the Matter Lab (Alan Aspuru Guzik group) and the Vector institute Toronto.

The [LabPics dataset](https://zenodo.org/record/3697452) was made by Mor Bismuth.


