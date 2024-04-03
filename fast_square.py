import numba
import numpy as np
import cv2 as cv


@numba.njit()
def calculate_intersection_area_over_union_area(x, y, s):
    # Assume the keypoints are squares! (keypoint.size is the side length of the square.)

    num_keypoints = len(x)
    iou_matrix = np.zeros((num_keypoints, num_keypoints))

    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            x0, y0, s0 = x[i], y[i], s[i]
            x1, y1, s1 = x[j], y[j], s[j]

            x0_min, x0_max = x0 - s0 / 2, x0 + s0 / 2
            y0_min, y0_max = y0 - s0 / 2, y0 + s0 / 2
            x1_min, x1_max = x1 - s1 / 2, x1 + s1 / 2
            y1_min, y1_max = y1 - s1 / 2, y1 + s1 / 2

            intersection_area = max(0, min(x0_max, x1_max) - max(x0_min, x1_min)) * max(0, min(y0_max, y1_max) - max(y0_min, y1_min))

            area0 = s0 ** 2
            area1 = s1 ** 2
            union_area = area0 + area1 - intersection_area

            iou = intersection_area / union_area
            iou_matrix[j, i] = iou

    return iou_matrix


def perform_nms(kp: list[cv.KeyPoint], threshold: float):
    #r = np.array([keypoint.size / 2 for keypoint in kp])
    s = np.array([keypoint.size for keypoint in kp])
    x = np.array([keypoint.pt[0] for keypoint in kp])
    y = np.array([keypoint.pt[1] for keypoint in kp])
    iou = calculate_intersection_area_over_union_area(x, y, s)

    kept_indices = []
    for i in range(len(kp)):
        if np.all(iou[i, kept_indices] <= threshold):  # kept_indices yerine :i kullanabiliriz yaklaşık sonuçlar için. Daha hızlı olur. Ama aynı şey değil!
            kept_indices.append(i)
    return [kp[i] for i in kept_indices]
