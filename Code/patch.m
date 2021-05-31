%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Program:To extract the patches of Water, Agriculture land and Bareland after annotation 
Input: Mean image of Water from S1 image (S1_Water.jpg)
       Mean image of Agriculture land from S1 image (S1_Agriculture_Land.jpg)
       Mean image of Bareland from S1 image (S1_Bareland.jpg)
       Mean image of Water from S2 image (S2_Water.jpg)
       Mean image of Agriculture land from S2 image (S2_Agriculture_Land.jpg)
       Mean image of Bareland from S2 image (S1_Bareland.jpg)
Output: 5 X 5 patches generated from input images and stored in folders (S1_Agriculture_Patches.jpg, S1_Water_Patches.jpg, 
        S1_Bareland_Patches.jpg, S2_Agriculture_Patches.jpg, S2_Water_Patches.jpg, S2_Bareland_Patches.jpg)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Extracting the patches
img=imread('S1_Agriculture_Land.jpg');
dd=0;
for ii=1:5:size(img,1)
    for jj=1:5:size(img,2)
        imm=img(ii:ii+4,jj:jj+4);
        imwrite(imm,['S1_Agriculture_Patches.jpg/',num2str(dd),'.jpg']);
        dd=dd+1;
    end
end
