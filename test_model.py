'''
Test model on 300W-LP dataset.
'''

import dlib
import cv2

if __name__ == '__main__':
    predictor_path = 'model/predictor_helen.dat'
    predictor = dlib.shape_predictor(predictor_path)

    f = open('image.txt', 'r')
    lines = f.readlines()

    i = 0
    while i < len(lines):
        image_path = lines[i].strip('\n')
        print('Image path : {}'.format(image_path))
        splits = image_path.split('/')
        dir = splits[0]
        path = splits[1]
        splits = path.split('.')
        name, suffix = splits[0], splits[1]

        image = cv2.imread(image_path)
        l = lines[i+1].strip('\n')
        position = l.split(' ')
        left = int(position[0])
        top = int(position[1])
        right = int(position[2])
        bottom = int(position[3])


        bounding_box = dlib.rectangle(left, top, right, bottom)

        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.putText(image, 'helen', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        shape = predictor(image, bounding_box)
        for j in range(68):
            x = shape.part(j).x
            y = shape.part(j).y
            cv2.circle(image, (x, y), 1, (0, 255, 255), 2)

        cv2.imwrite('output/{}/{}_helen.jpg'.format(dir, name), image)

        i += 2

    f.close()
