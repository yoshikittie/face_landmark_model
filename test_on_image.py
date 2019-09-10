import cv2
import dlib
import cv2
import os
import numpy as np
from head_pose_estimation import head_pose_estimtaion


if __name__ == '__main__':

    predictor_path = 'model/predictor.dat'
    predictor = dlib.shape_predictor(predictor_path)

    object_points = np.array([
        (5.311432, 5.485328, 3.987654),  # left eye, left corner
        (1.789930, 5.393625, 4.413414),  # left eye, right corner
        (-1.789930, 5.393625, 4.413414),  # right eye, left corner
        (-5.311432, 5.485328, 3.987654),  # right eye, right corner
        (0.000000, 0.000000, 6.763430),  # nose tip
        (2.774015, -2.080775, 5.048531),  # left mouth
        (-2.774015, -2.080775, 5.048531),  # right mouth
        (0.000000, -7.415691, 4.070434),  # chin
    ], dtype=np.float32)


    feature_points_index = [36, 39, 42, 45, 33, 48, 54, 8]  # eight points

    for root, dirs, files in os.walk('test/input'):
        for file in files:
            image = cv2.imread(os.path.join(root, file))
            img = head_pose_estimtaion(image, predictor, object_points, feature_points_index, [], [], [], [])
            splits = file.split('.')
            cv2.imwrite('test/output/{}.jpg'.format(splits[0]), img)

