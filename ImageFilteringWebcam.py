# External Libraries
import cv2
import numpy as np
import time
import psutil
import multiprocessing

# Global variables
# Start of performance counter
time_1 = time.perf_counter()

# Video capture of webcam (first webcam)
web_cam = cv2.VideoCapture(0)

# 'XVID' compresses video format.
compress = cv2.VideoWriter_fourcc(*'XVID')

# (1 Output of file 'output.avi'
# (2 calling fourcc variable
# (3 frames-per-second (FPS)
# (4 Frame size
output = cv2.VideoWriter('output.avi', compress, 20.0, (640, 480))

# Subtracts background
background_extractor = cv2.createBackgroundSubtractorMOG2()


class HardwareInfo:
    def __init__(self):
        pass

    def cpu_info(self):
        print("Physical cores", psutil.cpu_count(logical=False))
        print("Physical cores logical cores", psutil.cpu_count(logical=True))

        print("Usage Per Core:")
        for cores, per in enumerate(psutil.cpu_percent(percpu=True)):
            print(f"Core {cores}: {per}%")
        print(f"Usage: {psutil.cpu_percent()}%")


class ImageFiltering:
    def __init__(self):
        pass

    def original(self):
        # Displays image in window
        cv2.imshow('Original', frame)

    def black_and_white(self):
        # BGR gray conversion
        black_white = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('black/white', black_white)

    def record(self):
        # Calls global variable out. Takes frame as argument.
        output.write(frame)

    def colour_filtering(self):
        # hsv hue sat value
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Colour extraction
        # Converts green objects into a white mask.
        white = cv2.inRange(hsv, np.array([40, 100, 100]), np.array([80, 255, 255]))

        # Colour filters to only show red
        # Parameters, source 1, source 2 and mask.
        result = cv2.bitwise_and(frame, frame, mask=white)

        # Blurs the red filter
        # src = res. ksize = kernel size.
        blur = cv2.GaussianBlur(result, (15, 15), 0)

        # Displays to screen
        cv2.imshow('mask', white)
        cv2.imshow('result', result)
        cv2.imshow('blur', blur)

    def edges(self):

        # Image Gradients
        # CV_64F data type
        l = cv2.Laplacian(frame, cv2.CV_64F)

        # Ksize is kernel size, 5x5 regions of height and width.
        # Vertical
        s_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)

        # Horizontal
        s_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

        # frame, minVal, MaxVal.
        edges = cv2.Canny(frame, 100, 200)

        # Displays to screen
        cv2.imshow('laplacian', l)
        cv2.imshow('sobelx', s_x)
        cv2.imshow('sobely', s_y)
        cv2.imshow('edges', edges)

    def motion_sensor(self):
        # Applies subtraction to the frame
        motion = background_extractor.apply(frame)

        # Displays to screen
        cv2.imshow('fg', motion)


if __name__ == '__main__':
    # Testing time it takes to run
    time_2 = time.perf_counter()
    print(f'Finished in {time_2 - time_1} seconds')
    while True:

        # Gets clock information
        start_time = time.time()

        # Instantiates class
        i = ImageFiltering()

        # Instantiates class
        f = HardwareInfo()

        # Reads the capture, bool value.
        _, frame = web_cam.read()

        # Calling methods
        i.original()
        i.black_and_white()
        i.colour_filtering()
        i.edges()
        i.motion_sensor()
        i.record()

        # Pressing q exits web cam and breaks out of loop
        if cv2.waitKey(1) & 0xFF == ord('b'):
            break
        print("FPS: ", 1.0 / (time.time() - start_time))  # FPS = 1 / time to process loop
        # Call fps once per frame

    # Testing frames per second of the display

    f.cpu_info()
# Closes the application
output.release()
web_cam.release()

# Destroys the display
cv2.destroyAllWindows()
