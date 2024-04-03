import numba
import numpy as np
import cv2 as cv


@numba.njit()
def calculate_intersection_area_over_union_area(x, y, r):
    num_keypoints = len(x)
    iou_matrix = np.zeros((num_keypoints, num_keypoints))

    for i in range(num_keypoints):
        for j in range(i + 1, num_keypoints):
            x0, y0, r0 = x[i], y[i], r[i]
            x1, y1, r1 = x[j], y[j], r[j]

            distance = np.hypot(abs(x0 - x1), abs(y0 - y1))

            if distance > r0 + r1:
                intersection_area = 0
            elif distance <= (r0 - r1) and r0 >= r1:
                intersection_area = np.pi * r1 * r1
            elif distance <= (r1 - r0) and r1 >= r0:
                intersection_area = np.pi * r0 * r0
            else:
                alpha = np.arccos(((r0 * r0) + (distance * distance) - (r1 * r1)) / (2 * r0 * distance)) * 2
                beta = np.arccos(((r1 * r1) + (distance * distance) - (r0 * r0)) / (2 * r1 * distance)) * 2
                a1 = (0.5 * beta * r1 * r1 ) - (0.5 * r1 * r1 * np.sin(beta))
                a2 = (0.5 * alpha * r0 * r0) - (0.5 * r0 * r0 * np.sin(alpha))
                intersection_area = a1 + a2

            union_area = np.pi * r0 ** 2 + np.pi * r1 ** 2 - intersection_area
            iou = intersection_area / union_area
            iou_matrix[j, i] = iou

    return iou_matrix


def perform_nms(kp: list[cv.KeyPoint], threshold: float):
    r = np.array([keypoint.size / 2 for keypoint in kp])  # Divided by 2 because the size is the diameter.
    x = np.array([keypoint.pt[0] for keypoint in kp])
    y = np.array([keypoint.pt[1] for keypoint in kp])
    iou = calculate_intersection_area_over_union_area(x, y, r)

    kept_indices = []
    for i in range(len(kp)):
        if np.all(iou[i, kept_indices] <= threshold):  # kept_indices yerine :i kullanabiliriz yaklaşık sonuçlar için. Daha hızlı olur. Ama aynı şey değil!
            kept_indices.append(i)
    return [kp[i] for i in kept_indices]
