import cv2
import numpy as np


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    
    current_mode = 'o'
    sigma = 1.0

    print("Program Started. Controls:")
    print(" o: Original | x: Sobel X | y: Sobel Y | m: Magnitude")
    print(" s: Sobel+Threshold | l: LoG | +/-: Sigma Adjust | q: Quit")

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break


        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)


        output_img = None


        if current_mode == 'o':
            output_img = frame

        elif current_mode == 'x':
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            output_img = cv2.convertScaleAbs(sobelx)

        elif current_mode == 'y':
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            output_img = cv2.convertScaleAbs(sobely)

        elif current_mode == 'm':
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            output_img = cv2.convertScaleAbs(magnitude)

        elif current_mode == 's':
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            mag_uint8 = cv2.convertScaleAbs(magnitude)
            _, output_img = cv2.threshold(mag_uint8, 70, 255, cv2.THRESH_BINARY)

        elif current_mode == '1':

            laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
            output_img = cv2.convertScaleAbs(laplacian)


        if len(output_img.shape) == 2:
            display_img = cv2.cvtColor(output_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = output_img.copy()

        info_text = f"Mode: {current_mode} | Sigma: {sigma:.1f}"
        cv2.putText(display_img, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Lab 5 Task 1', display_img)


        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('o'):
            current_mode = 'o'
        elif key == ord('x'):
            current_mode = 'x'
        elif key == ord('y'):
            current_mode = 'y'
        elif key == ord('m'):
            current_mode = 'm'
        elif key == ord('s'):
            current_mode = 's'
        elif key == ord('1'):
            current_mode = '1'
        elif key == ord('+') or key == ord('='):
            sigma += 0.5
            print(f"Sigma increased to: {sigma}")
        elif key == ord('-') or key == ord('_'):
            if sigma > 0.5:
                sigma -= 0.5
                print(f"Sigma decreased to: {sigma}")


    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()