import cv2
import os


def valid(value, v1, v2):
    return value >= v1 and value <= v2


def face_detection(img, model):
    conf_thres = 0.9
    width = img.shape[1]
    height = img.shape[0]

    face_blob = cv2.dnn.blobFromImage(img, 1, (width, height), [104, 117, 123])
    model.setInput(face_blob)
    faces = model.forward()

    face_location = []
    for i in range(faces.shape[2]):
        conf = faces[0, 0, i, 2]
        x1 = int(faces[0, 0, i, 3] * width)
        y1 = int(faces[0, 0, i, 4] * height)
        x2 = int(faces[0, 0, i, 5] * width)
        y2 = int(faces[0, 0, i, 6] * height)

        f = valid(x1, 0, width) and valid(y1, 0, height) and valid(x2, 0, width) and valid(y2, 0, height)
        if not f:
            continue
        if conf >= conf_thres:
            face_location.append([x1, y1, x2, y2])

    return face_location


if __name__ == '__main__':

    model = cv2.dnn.readNetFromTensorflow('lib/opencv_face_detector_uint8.pb', 'lib/opencv_face_detector.pbtxt')

    f = open('HELEN.txt', 'r')
    out = open('HELEN.xml', 'w')

    out.write('<dataset>\n')
    out.write('<name> Training faces </name>\n')
    out.write('<comment> </comment>\n')
    out.write('<images>\n')
    # out = open('image.txt', 'a')

    lines = f.readlines()

    i = 0

    while i < len(lines):
        if i % 69 == 0:
            l = lines[i].strip('\n')
            i1 = l.find('/')
            i2 = l.rfind('_')
            img_path = l[i1 + 1:i2] + '.jpg'
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                face_location = face_detection(img, model)
                if len(face_location) == 1:
                    # out.write('{}\n'.format(img_path))
                    # out.write('{} {} {} {}\n'.format(face_location[0][0], face_location[0][1], face_location[0][2], face_location[0][3]))
                    out.write('<image file = \'{}\'>\n'.format(img_path))
                    top = face_location[0][1]
                    left = face_location[0][0]
                    width = face_location[0][2] - face_location[0][0]
                    height = face_location[0][3] - face_location[0][1]
                    out.write(
                         '<box top = \'{}\' left = \'{}\' width = \'{}\' height = \'{}\'>\n'.format(top, left, width,
                                                                                                  height))
                    for j in range(i + 1, i + 69):
                         l = lines[j].strip('\n')
                         splits = l.split(' ')
                         out.write('<part name = \'{:0>2d}\' x = \'{}\' y = \'{}\'/>\n'.format(j - i - 1, splits[0], splits[1]))
                    out.write('</box>\n')
                    out.write('</image>\n')
                # else:
                #     print('File name : {} Face number : {}'.format(img_path, len(face_location)))
                #     cv2.imwrite('img/other/{}'.format(img_path), img)
            i += 69
    # out.close()

    out.write('</images>\n')
    out.write('</dataset>\n')
    out.close()
