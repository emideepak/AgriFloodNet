%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Program:To Take the Mean of the Water specific bands Image bands 
Input: Bands for Sentinel-2 for flood (S2_2019-03-22_B02_band_1.png,S2_2019-03-22_B08_band_1.png and S2_2019-03-22_B11_band_1.png)
       Bands for Sentinel-2 for No flood (S2_2019-02-22_B02_band_1.png,S2_2019-02-22_B08_band_1.png and S2_2019-02-22_B11_band_1.png)
       Bands for Sentinel-1 for flood (S1_2019-03-22_VH.png and S1_2019-03-22_VH.png )
       Bands for Sentinel-1 for flood (S1_2019-02-22_VH.png and S1_2019-02-22_VH.png )
Output: Mean image of Sentinel-2 Flood (S2_FloodMean.png)
        Mean image of Sentinel-2 Flood (S2_NoFloodMean.png)
        Mean image of Sentinel-1 Flood (S1_FloodMean.png)
        Mean image of Sentinel-1 Flood (S1_FloodMean.png)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import cv2 
import matplotlib.pyplot as plt
from skimage.io import imread
import numpy as np

#Mean for NoFlood and Flood in Sentinel-2
#Taking the mean of Three image
img1=imread("/content/S2_2019-03-22_B02_band_1.png");
img1 = img1.astype(np.float64)
img2=imread("/content/S2_2019-03-22_B08_band_1.png");
img2= img2.astype(np.float64)
img3=imread("/content/S2_2019-03-22_B11_band_1.png");
img3= img3.astype(np.float64)
img4=(img1+img2+img3)/3.0;
import numpy as np
import imageio

# suppose that img's dtype is 'float64'
img_uint8 = img4.astype(np.uint8)
# and then
imageio.imwrite('S2_NoFloodMean.png', img_uint8)

#Mean for NoFlood and Flood in Sentinel-1
#Taking the mean of Two image
img1=imread("/content/VH.png");
img1 = img1.astype(np.float64)
img2=imread("/content/VV.png");
img2= img2.astype(np.float64)
img3=(img1+img2)/2.0;
import numpy as np
import imageio

# suppose that img's dtype is 'float64'
img_uint8 = img3.astype(np.uint8)
# and then
imageio.imwrite('S1_FloodMean.png', img_uint8)
