from PIL import Image
import numpy as np

start_string = "Spr_1b_"
image_file_ex = ".png"

for i in range(3):
    #the + 1 is because we dont have a Spr_1b_000.png
    image_string = start_string + str(i + 1).zfill(3)
    im = Image.open(image_string + image_file_ex)
    im = im.convert("RGBA")
    output_image = []
    for p in list(im.getdata()):
        output_image.append(float(p[0]) / 255.0)
        output_image.append(float(p[1]) / 255.0)
        output_image.append(float(p[2]) / 255.0)
        output_image.append(float(p[3]) / 255.0)
    np.save("ml_data/" + image_string, np.array(output_image))
    print(len(output_image))