import copy
import cv2 as cv

from kuwahara_filter import kuwahara_filter


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