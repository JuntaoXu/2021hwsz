def annotation_json_to_txt(xmin, ymin, xmax, ymax, pic_width, pic_height):
    x = (xmin + xmax) / (2 * pic_width)
    y = (ymin + ymax) / (2 * pic_width)
    w = (xmax - xmin)/pic_width
    h = (ymax - ymin)/pic_height
    return x, y, w, h


def annotation_txt_to_json(x, y, w, h, pic_width, pic_height):
    xmin = (x - (w / 2)) * pic_width
    xmax = (x + (w / 2)) * pic_width
    ymin = (y - (h / 2)) * pic_height
    ymax = (y + (h / 2)) * pic_height
    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    # test code
    xmin = 100
    xmax = 200
    ymin = 100
    ymax = 200
    pic_width = 1000
    pic_height = 1000

    x, y, w, h = (annotation_json_to_txt(xmin, ymin, xmax, ymax, pic_width, pic_height))

    print(annotation_txt_to_json(x, y, w, h, pic_width, pic_height))
