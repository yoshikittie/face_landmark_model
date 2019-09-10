import dlib
import cv2
import os
import numpy as np
from utils import draw_axis, rotation_matrix_to_euler_angle
from csv import reader


def head_pose_estimtaion(image, predictor, object_points, feature_points_index, camera_matrix, dist_coeffs, offset_x,
                         offset_y):

    feature_points = []

    size = image.shape

    print('Original shape : {} {}'.format(size[0], size[1]))

    factor = size[1] / size[1]

    image_resized = image

    face_detector = dlib.get_frontal_face_detector()

    dets = face_detector(image_resized, 1)

    for k, d in enumerate(dets):

        size = image_resized.shape

        image_points = predictor(image_resized, d)

        for i in feature_points_index:
            x = image_points.part(i).x
            y = image_points.part(i).y
            feature_points.append([float(x), float(y)])
            cv2.circle(image_resized, (x, y), 1, (0, 255, 255), 2)

        feature_points = np.array(feature_points, dtype=np.float32)

        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ])
        dist_coeffs = np.zeros((4, 1))

        object_points = object_points.reshape((1, object_points.shape[0], object_points.shape[1]))
        feature_points = feature_points.reshape((1, feature_points.shape[0], feature_points.shape[1]))

        # _, cm, dist, _, _ = cv2.calibrateCamera(object_points, feature_points, (size[0], size[1]), camera_matrix, dist_coeffs, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

        _, rotation_vector, translation_vector = cv2.solvePnP(object_points, feature_points, camera_matrix, dist_coeffs,
                                                              flags=cv2.SOLVEPNP_ITERATIVE)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_matrix)
        euler_angle2 = rotation_matrix_to_euler_angle(rotation_matrix)

        cv2.putText(image_resized, 'Pitch : {:.2f}'.format(euler_angle[0, 0]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # print('Euler angle : {}'.format(euler_angle))
        draw_axis(image_resized, euler_angle2)

        return image_resized

    # cv2.imshow('face', image)
    #
    # if cv2.waitKey(0) == ord('q'):
    #     cv2.destroyAllWindows()


if __name__ == '__main__':

    predictor_path = 'model/predictor.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # feature_points_index = [36, 45, 33, 48, 54] # five points

    # object_points = np.array([
    #    (5.311432, 5.485328, 3.987654), # left eye, left corner
    #    # (1.789930, 5.393625, 4.413414), # left eye, right corner
    #    # (-1.789930, 5.393625, 4.413414), # right eye, left corner
    #    (-5.311432, 5.485328, 3.987654), # right eye, right corner
    #    (0.000000, 0.000000, 6.763430), # nose tip
    #    (2.774015, -2.080775, 5.048531), # left mouth
    #    (-2.774015, -2.080775, 5.048531), # right mouth
    #   # (0.000000, -7.415691, 4.070434), # chin
    # ])

    #object_points = np.array([
    #    (5.311432, 5.485328, 3.987654),  # left eye, left corner
    #    (1.789930, 5.393625, 4.413414),  # left eye, right corner
    #    (-1.789930, 5.393625, 4.413414),  # right eye, left corner
    #    (-5.311432, 5.485328, 3.987654),  # right eye, right corner
    #    (0.000000, 0.000000, 6.763430),  # nose tip
    #    (2.774015, -2.080775, 5.048531),  # left mouth
    #    (-2.774015, -2.080775, 5.048531),  # right mouth
    #    (0.000000, -7.415691, 4.070434),  # chin
    #], dtype=np.float32)

    #object_points = object_points * 100

    #feature_points_index = [36, 39, 42, 45, 33, 48, 54, 8]  # eight points

    object_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])

    feature_points_index = [33, 8, 36, 45, 48, 54]

    image = cv2.imread('image/test7.jpg')
    img = head_pose_estimtaion(image, predictor, object_points, feature_points_index, [], [],
                                   [], [])
    cv2.imwrite('image/test7_out.jpg', img)



