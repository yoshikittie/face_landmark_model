import cv2
import numpy as np
from math import cos, sin, atan2

def is_rotation_matrix(R):
    should_be_indentity = np.dot(R.T, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - should_be_indentity)
    return n < 1e-6

def normalize(theta):
    if theta > 90:
        ret = theta - 180
    elif theta < -90:
        ret = theta + 180
    else:
        ret = theta

    return ret

def rotation_matrix_to_euler_angle(R):
    assert (is_rotation_matrix(R))
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    factor = 180 / np.pi

    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0

    x = normalize(x * factor)
    y = normalize(y * factor)
    z = normalize(z * factor)

    return x, y, z


def draw_axis(image, pose):
    pitch = -pose[0] * np.pi / 180
    yaw = -pose[1] * np.pi / 180
    roll = pose[2] * np.pi / 180
    
    size = image.shape
    origin_x = size[1] / 2
    origin_y = size[0] / 2
    factor = 100
    
    # X-axis, drawn in red. pitch
    
    x1 = factor * (cos(yaw) * cos(roll)) + origin_x
    y1 = factor * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + origin_y

    # Y-axis, drawn in green, yaw
    
    x2 = factor * (-cos(yaw) * sin(roll)) + origin_x
    y2 = factor * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + origin_y
    
    # Z-axis, drawn in blue, roll
    
    x3 = factor * (sin(yaw)) + origin_x
    y3 = factor * (-cos(yaw) * sin(pitch)) + origin_y
    
    cv2.line(image, (int(origin_x), int(origin_y)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(image, (int(origin_x), int(origin_y)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(image, (int(origin_x), int(origin_y)), (int(x3), int(y3)), (255, 0, 0), 3)
    

def plot(filename):
    image = cv2.imread(filename)
    cv2.imshow('img', image)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

def ground_truth(filename):

    f = open(filename, 'r')
    lines = f.readlines()
    i = 0
    while i < len(lines):
        l = lines[i].strip('\n')
        i1 = l.find('/')
        i2 = l.rfind('_')
        image_path = l[i1 + 1 : i2]
        print(image_path)
        image = cv2.imread('{}.jpg'.format(image_path))
        for j in range(i + 1, i + 69):
            l = lines[j].strip('\n')
            position = l.split(' ')
            x = int(position[0])
            y = int(position[1])
            cv2.circle(image, (x, y), 1, (0, 255, 255), 2)
        cv2.imwrite('output/{}_ground.jpg'.format(image_path), image)
        i += 69

if __name__ == '__main__':
    ground_truth('LFPW.txt')