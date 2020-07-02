import numpy as np
import cv2
import imutils
import time


def main():

    pen_img = cv2.resize(cv2.imread('pen.png',1), (50, 50))
    eraser_img = cv2.resize(cv2.imread('eraser.png',1), (50, 50))

    # define the lower and upper boundaries of the colors in HSV color space
    lower_range = np.array([110,90,50])
    upper_range = np.array([130,255,255])
    
    Kernal = np.ones((5,5), np.uint8)
    
    # Threshold for noise
    noiseth = 80
    
    x1, y1 = 0, 0

    wiper_thresh = 40000

    background_threshold = 600

    # With this variable we will monitor the time between previous switch.
    last_switch = time.time()
    
    switch = 'Pen'

    clear = False
    
    thickness = 2
    
    # Create a background subtractor Object
    backgroundobject = cv2.createBackgroundSubtractorMOG2(detectShadows = False)
    
    cap = cv2.VideoCapture(0)
    canvas = None

    while True:
        # Grab the current frame
        _, frame = cap.read()
        
        # Flip horizontally
        frame = cv2.flip( frame, 1 )

        if canvas is None:
            canvas = np.zeros_like(frame)
	
        # Take the top left of the frame and apply the background subtractor
        top_left = frame[50: 100, 0: 50]
        fgmask = backgroundobject.apply(top_left)
        
        switch_thresh = np.sum(fgmask==255)

        if switch_thresh>background_threshold and (time.time()-last_switch) > 1:
             last_switch = time.time()
             
             if switch == 'Pen':
                 switch = 'Eraser'

             else:
                 switch = 'Pen'

	# Blur the frame
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
       
        # Convert to the HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Construct masks
        mask = cv2.inRange(hsv, lower_range, upper_range)
        
        # Perform morphological operations to get rid of the noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, Kernal) 
        mask = cv2.dilate(mask, Kernal, iterations=2)
        
        # Find contours in the mask
        cnts, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        # Only proceed if at least one contour is found and area greater than noise threshold
        if cnts and cv2.contourArea(max(cnts,key = cv2.contourArea)) > noiseth:
            # Find the largest contour in the mask
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            x2, y2 = center
            
            #thickness depends on the area of circle enclosing the contour
            thickness=int(area/500)
            if thickness < 2:
              thickness=2
            
            if x1 == 0 and y1 == 0:
                x1, y1 = x2, y2
            else:
                if switch == 'Pen':
                     # Draw the line on the canvas
                     canvas = cv2.line(canvas, (x1, y1), (x2, y2), [255,0,0], thickness)

                else:
                     cv2.circle(canvas, (x2, y2), thickness,(0,0,0), -1)
            
            x1, y1, = x2, y2

            # Puts circle at center of marker, the point where the drawing will be done
            cv2.circle(frame, center, thickness, (0, 255, 255), -1)

            # If the marker is close enough to the screen, it clears it
            if area > wiper_thresh:
                cv2.putText(canvas, 'Clearing Canvas', (100, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA)
                clear = True
        else:
            x1, y1 = 0, 0

        # Put canvas on top of the frame
        _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
        background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(foreground, background)
        mask_ = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        if switch != 'Pen':
             frame[50: 100, 0: 50] = eraser_img
        else:
             frame[50: 100, 0: 50] = pen_img

        cv2.imshow('Trackbars',frame)
        
	# If q is pressed, video stops
        close_key = cv2.waitKey(1) & 0xFF
        if close_key == ord("e"):
            break
        if clear:
            time.sleep(1)
            canvas = None
            clear = False

    # When loop ends, exit program
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()