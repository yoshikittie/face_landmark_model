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

    # image_resized = cv2.resize(image, (int(factor * size[1]), int(factor * size[0])), 0.0, 0.0, interpolation=cv2.INTER_CUBIC)

    size = image_resized.shape

    print('Resized shape : {} {}'.format(size[0], size[1]))

    bounding_box = dlib.rectangle(0, 0, size[1], size[0])
    image_points = predictor(image_resized, bounding_box)

    for i in feature_points_index:
        x = image_points.part(i).x
        y = image_points.part(i).y
        feature_points.append([float(x), float(y)])
        cv2.circle(image_resized, (x, y), 1, (0, 255, 255), 2)

    feature_points = np.array(feature_points, dtype=np.float32)

    focal_length = size[1]
    center = (size[1] / 2, size[1] / 2)
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

    predictor_path = 'model/predictor_afw.dat'
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

    # object_points = object_points * 100

    feature_points_index = [36, 39, 42, 45, 33, 48, 54, 8]  # eight points

    face_location_file = reader(open('face_location.csv', encoding='utf-8'))
    mtx_file = reader(open('mtx.csv', encoding='utf-8'))
    dist_file = reader(open('dist.csv', encoding='utf-8'))

    face_location = []
    for row in face_location_file:
        r = list(map(int, row))
        face_location.append(r)

    camera_matrix = []
    for i, row in enumerate(mtx_file):
        r = list(map(float, row))
        camera_matrix.append(r)
    camera_matrix = np.array(camera_matrix)

    dist_coeffs = []
    for i, row in enumerate(dist_file):
        r = list(map(float, row))
        dist_coeffs.append(r)
    dist_coeffs = np.array(dist_coeffs)

    # for root, dirs, files in os.walk('face'):
    # for file in files:
    # image = cv2.imread(os.path.join(root, file))
    # print(file)
    # img = head_pose_estimtaion(image, predictor, object_points, feature_points_index, camera_matrix, dist_coeffs, )
    # splits = file.split('.')
    # cv2.imwrite('face_output/{}_{}.jpg'.format(splits[0], 'helen'), img)

    for i, location in enumerate(face_location):
        image = cv2.imread('face/face_{}.jpg'.format(i))
        img = head_pose_estimtaion(image, predictor, object_points, feature_points_index, camera_matrix, dist_coeffs,
                                   location[0], location[1])
        cv2.imwrite('face_output/face_{}_{}.jpg'.format(i, 'afw'), img)
