# AgriFloodNet Architecture
Authors: A. Emily Jenifer, A. Aparna , Sudha Natarajan and Anil Kumar.

AgriFloodNet: A dual CNN architecture for crop damage assessment due to flood using bi-temporal multi-sensor images

## Cloning the repository

Clone the git repository by running git bash in your computer and run the following command

`https://github.com/emideepak/AgriFloodNet.git`

or click on the download button and extract the zip file

## Installing dependencies

On a PC running Windows,

Install `MATLAB` with Image Processing toolbox from `https://www.mathworks.com/downloads/`.

Install `Anaconda` from `https://www.anaconda.com/products/individual` and run the following commands in your conda prompt.

`conda install tensorflow keras opencv`

`import tensorflow`

`import keras`

`import cv2`

`pip install pixelmatch`

`conda install imutils scikit-learn`

## Execution of codes 

1. Run the Mean.py python code on Anaconda

2. Run patch.m on MATLAB

3. Run Spilting.py python code on Anaconda

4. Run AgriFloodNet_CNN.py python code on Anaconda

5. Run Training.py python code on Anaconda

6. Run Testing.py python code on Anaconda

7. Run the DecisionFusion.py python code on Anaconda

8. The above steps are executed for both before and after Flood in Sentinel 1, 2 images.

9. Run ChangeDetection.py python code on Anaconda
The final change map is stored in `ChangeMap.png`




