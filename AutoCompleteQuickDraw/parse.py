from PIL import Image, ImageDraw
import numpy as np
import jsonlines, math, time, sys
import scipy.interpolate as interp

def resize_interp(vect, dim):
    vect_ref = np.zeros(dim)
    vect_interp = interp.interp1d(np.arange(vect.size), vect)
    vect_interp_ref = vect_interp(np.linspace(0, vect.size - 1, vect_ref.size))
    return vect_interp_ref

canvas_size_x = 2000
canvas_size_y = 2000

outputs = []
cur_img_counter = 0
counter = 0

ndjson = jsonlines.open("house.ndjson")
for drawing in ndjson:

    if cur_img_counter > 9:
        break

    key_id = drawing["key_id"]
    all_strokes = drawing["drawing"]
    im = Image.new('RGBA', (canvas_size_x, canvas_size_y), (0, 0, 0, 0))

    for stroke in all_strokes:
        stroke_x_coords = stroke[0]
        stroke_y_coords = stroke[1]

        for i in range(len(stroke_x_coords) - 1):
            draw = ImageDraw.Draw(im)
            draw.line((stroke_x_coords[i], stroke_y_coords[i], stroke_x_coords[i + 1], stroke_y_coords[i + 1]), fill=(0, 0, 0, 255), width=3)

    x_max = 0
    y_max = 0
    x_min = 99999
    y_min = 99999

    start = time.time()

    for x in range(canvas_size_x):
        for y in range(canvas_size_y):
            r, g, b, a = im.getpixel((x, y))
            if a > 0:
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y

    end = time.time()

    print("crop hints: " + str(x_min) + ", " + str(y_min) + " " + str(x_max) + ", " + str(y_max) + " elapsed: " + str(end - start))
    im = im.crop((x_min, y_min, x_max, y_max))

    width, height = im.size

    max_dim = 0
    if width > height:
        max_dim = width
    else:
        max_dim = height

    x1 = int(math.floor((max_dim - width) / 2))
    y1 = int(math.floor((max_dim - height) / 2))

    newImage = Image.new('RGBA', (max_dim, max_dim), (0, 0, 0, 0))
    newImage.paste(im, (x1, y1, x1 + width, y1 + height))
    newImage.resize((200, 200)).save("./img/" + str(cur_img_counter) + ".png")

    img_array = np.zeros((200, 200), dtype=np.int16)
    try:
        for x in range(200):
            for y in range(200):
                r, g, b, a = newImage.getpixel((x, y))
                img_array[x][y] = a
    except:
        cur_img_counter -= 1
        continue

    # convolutions should be 50px x 50px
    img_array = img_array.flatten()

    # full img save
    '''
    img_array_full_50 = resize_interp(img_array, 2500)
    img_array_full_50_name = str(counter) + "_" + key_id + ".txt"
    np.savetxt("train/arr/" + img_array_full_50_name, img_array_full_50)
    outputs.append(cur_img_counter)
    counter += 1
    '''

    # 50 by 50 regions
    for i in range(16):
        img_arr_part = img_array[i * 2500 : (i + 1) * 2500]
        img_arr_part_name = str(counter) + "_" + key_id + ".txt"
        np.savetxt("train/arr/" + img_arr_part_name, img_arr_part)
        outputs.append(cur_img_counter)
        counter += 1

    cur_img_counter += 1

outputs_arr = np.array(outputs)
np.savetxt("train/outputs.txt", outputs_arr)
