def calculate_intersection_area_over_union_area(keypoint0, keypoint1):
    # Assume the keypoints are squares! (keypoint.size is the side length of the square.)

    x0, y0 = keypoint0.pt
    x1, y1 = keypoint1.pt
    s0 = keypoint0.size
    s1 = keypoint1.size

    # Calculate the intersection area.
    x0_min, x0_max = x0 - s0 / 2, x0 + s0 / 2
    y0_min, y0_max = y0 - s0 / 2, y0 + s0 / 2
    x1_min, x1_max = x1 - s1 / 2, x1 + s1 / 2
    y1_min, y1_max = y1 - s1 / 2, y1 + s1 / 2

    intersection_area = max(0, min(x0_max, x1_max) - max(x0_min, x1_min)) * max(0, min(y0_max, y1_max) - max(y0_min, y1_min))

    # Calculate the union area.
    area0 = s0 ** 2
    area1 = s1 ** 2
    union_area = area0 + area1 - intersection_area

    iou = intersection_area / union_area
    return iou


def perform_nms(kp, threshold):
    new_kp = [kp[0]]
    for candidate_keypoint in kp[1:]:
        for existing_keypoint in new_kp:
            if calculate_intersection_area_over_union_area(existing_keypoint, candidate_keypoint) > threshold:
                break
        else:
            new_kp.append(candidate_keypoint)
    return new_kp
