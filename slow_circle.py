from math import pi, acos, sin
import math
import numpy as np


# def calculate_intersection_area(distance, r0, r1):
#     """Return the area of intersection of two circles.

#     The circles have radii R and r, and their centres are separated by d.
#     """

#     if distance <= abs(r0 - r1):
#         # One circle is entirely enclosed in the other.
#         return np.pi * min(r0, r1) ** 2
#     if distance >= r1 + r0:
#         # The circles don't overlap at all.
#         return 0

#     r2, R2, d2 = r1 ** 2, r0 ** 2, distance ** 2
#     alpha = np.arccos((d2 + r2 - R2) / (2 * distance * r1 + 1e-6))
#     beta = np.arccos((d2 + R2 - r2) / (2 * distance * r0 + 1e-6))
#     return r2 * alpha + R2 * beta - 0.5 * (r2 * np.sin(2 * alpha) + R2 * np.sin(2 * beta))


def calculate_intersection_area(distance, r0, r1):
    """Return the area of intersection of two circles.

    The circles have radii R and r, and their centres are separated by d.
    """

    if distance > r0 + r1:
        ans = 0
 
    elif distance <= (r0 - r1) and r0 >= r1:
        ans = pi * r1 * r1
 
    elif distance <= (r1 - r0) and r1 >= r0:
        ans = pi * r0 * r0
 
    else:
        alpha = acos(((r0 * r0) + (distance * distance) - (r1 * r1)) / (2 * r0 * distance)) * 2
        beta = acos(((r1 * r1) + (distance * distance) - (r0 * r0)) / (2 * r1 * distance)) * 2
         
        a1 = (0.5 * beta * r1 * r1 ) - (0.5 * r1 * r1 * sin(beta))
        a2 = (0.5 * alpha * r0 * r0) - (0.5 * r0 * r0 * sin(alpha))
        ans = a1 + a2

    return ans


def calculate_intersection_area_over_union_area(keypoint0, keypoint1):
    # TODO: Bunun yerine alttaki _intersection_area_over_small_area olabilir her yerde!!!

    r0 = keypoint0.size / 2
    r1 = keypoint1.size / 2

    distance = math.hypot(abs(keypoint0.pt[0] - keypoint1.pt[0]), abs(keypoint0.pt[1] - keypoint1.pt[1]))

    intersection_area = calculate_intersection_area(distance, r0, r1)
    union_area = math.pi * r0 ** 2 + math.pi * r1 ** 2 - intersection_area
    return intersection_area / union_area


def perform_nms(kp, threshold):
    new_kp = [kp[0]]
    for candidate_keypoint in kp[1:]:
        for existing_keypoint in new_kp:
            if calculate_intersection_area_over_union_area(existing_keypoint, candidate_keypoint) > threshold:
                break
        else:
            new_kp.append(candidate_keypoint)
    return new_kp
