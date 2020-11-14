import copy
import numpy as np
import cv2 as cv


def kuwahara_filter(image, kernel_size=5):
    """桑原フィルターを適用した画像を返す

    Args:
        image: OpenCV Image
        kernel_size: Kernel size is an odd number of 5 or more

    Returns:
        Image after applying the filter.
    """
    height, width, channel = image.shape[0], image.shape[1], image.shape[2]

    r = int((kernel_size - 1) / 2)
    r = r if r >= 2 else 2

    image = np.pad(image, ((r, r), (r, r), (0, 0)), "edge")

    average, variance = cv.integral2(image)
    average = (average[:-r - 1, :-r - 1] + average[r + 1:, r + 1:] -
               average[r + 1:, :-r - 1] - average[:-r - 1, r + 1:]) / (r +
                                                                       1)**2
    variance = ((variance[:-r - 1, :-r - 1] + variance[r + 1:, r + 1:] -
                 variance[r + 1:, :-r - 1] - variance[:-r - 1, r + 1:]) /
                (r + 1)**2 - average**2).sum(axis=2)

    def filter(i, j):
        return np.array([
            average[i, j], average[i + r, j], average[i, j + r], average[i + r,
                                                                         j + r]
        ])[(np.array([
            variance[i, j], variance[i + r, j], variance[i, j + r],
            variance[i + r, j + r]
        ]).argmin(axis=0).flatten(), j.flatten(),
            i.flatten())].reshape(width, height, channel).transpose(1, 0, 2)

    filtered_image = filter(*np.meshgrid(np.arange(height), np.arange(width)))

    filtered_image = filtered_image.astype(image.dtype)
    filtered_image = filtered_image.copy()

    return filtered_image


def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.resize(frame, (960, 540))
        original_frame = copy.deepcopy(frame)

        frame = kuwahara_filter(frame, 5)

        cv.imshow('original', original_frame)
        cv.imshow('kuwabara', frame)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()