%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Program:To identify the changes due to flood on agriculture land
Input: DecisionFusion output after flood (Combined_After_map.png)
       DecisionFusion output before flood (Combined_Before_map.png)
Output: Change map of flood affected in agriculture land (Change_Map.png)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
from PIL import Image

from pixelmatch.contrib.PIL import pixelmatch

img_a = Image.open("Combined After map.png")
img_b = Image.open("Combined Before map.png")
img_diff = Image.new("RGBA", img_a.size)

# note how there is no need to specify dimensions
mismatch = pixelmatch(img_a, img_b, img_diff, includeAA=True)

img_diff.save("ChangeMap.png")