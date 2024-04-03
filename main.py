import random
import time
import cv2 as cv
import numpy as np

import fast_circle
import fast_square
import slow_square
import slow_circle


def generate_random_keypoint():
    x = random.randint(0, 800)
    y = random.randint(0, 800)
    size = random.randint(10, 50)
    angle = 0
    response = random.uniform(0, 1)
    return cv.KeyPoint(x, y, size, angle, response)


def show_keypoints(kp, title):
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    for keypoint in kp:
        cv.circle(img, (int(keypoint.pt[0]), int(keypoint.pt[1])), int(keypoint.size / 2), (255, 0, 0), 2)
    cv.imshow(title, img)
    cv.waitKey(0)


def main():
    random.seed(0)

    kp_count = 4096
    kp = [generate_random_keypoint() for _ in range(kp_count)]
    kp.sort(key=lambda keypoint: keypoint.response, reverse=True)
    threshold = 0.5

    start = time.time()
    kp_slow_circle = slow_circle.perform_nms(kp, threshold)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time for slow_circle.py: {elapsed_time:.2f} seconds")

    _ = fast_circle.perform_nms(kp, threshold)  # Numba needs to compile the function first.

    start = time.time()
    kp_fast_circle = fast_circle.perform_nms(kp, threshold)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time for fast_circle.py: {elapsed_time:.2f} seconds")   

    start = time.time()
    kp_slow_square = slow_square.perform_nms(kp, threshold)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time for slow_square.py: {elapsed_time:.2f} seconds")

    _ = fast_square.perform_nms(kp, threshold)  # Numba needs to compile the function first.

    start = time.time()
    kp_fast_square = fast_square.perform_nms(kp, threshold)
    end = time.time()
    elapsed_time = end - start
    print(f"Elapsed time for fast_square.py: {elapsed_time:.2f} seconds")

    print(f"(EXACT)  {len(kp_slow_circle)=}")
    print(f"(EXACT)  {len(kp_fast_circle)=}")
    print(f"(APPROX) {len(kp_slow_square)=}")
    print(f"(APPROX) {len(kp_fast_square)=}")

    # My conclusion: Square approximation is not necessary.

    # show_keypoints(kp, "Original keypoints")
    # show_keypoints(kp_slow_circle, "Keypoints after NMS (slow_circle.py)")
    # show_keypoints(kp_fast_circle, "Keypoints after NMS (fast_circle.py)")
    # show_keypoints(kp_slow_square, "Keypoints after NMS (slow_square.py)")
    # show_keypoints(kp_fast_square, "Keypoints after NMS (fast_square.py)")

    assert len(kp_slow_square) == len(kp_fast_square)
    for keypoint_slow, keypoint_fast in zip(kp_slow_square, kp_fast_square):
        assert keypoint_slow.pt == keypoint_fast.pt
        assert keypoint_slow.size == keypoint_fast.size
        assert keypoint_slow.response == keypoint_fast.response

    assert len(kp_slow_circle) == len(kp_fast_circle)
    for keypoint_slow, keypoint_fast in zip(kp_slow_circle, kp_fast_circle):
        assert keypoint_slow.pt == keypoint_fast.pt
        assert keypoint_slow.size == keypoint_fast.size
        assert keypoint_slow.response == keypoint_fast.response


if __name__ == "__main__":
    main()
