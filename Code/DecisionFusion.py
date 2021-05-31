##################################################################################
# Program: Generating overlaid images for qualitative analysis of patches 
# Inputs:  S1 ans S2 Images generated from AgrifloodNet 
# Output:  Combined before and after map of S1 and S2 images (Combined_Before_map.png)
###################################################################################
try:
    from PIL import Image
except ImportError:
    import Image

background = Image.open("S2_Before_ Flood.jpg")
overlay = Image.open("S1_BeforeFlood.jpg")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
new_img.save("Combined_Before_map.png","PNG")