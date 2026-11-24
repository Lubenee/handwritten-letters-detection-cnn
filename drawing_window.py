import cv2
import numpy as np
import constants
from helpers import image_to_emnist_format

class Window:
    def __init__(self, model):
        self.should_refresh = True
        self.should_predict = False
        self.frame = None
        self.model = model
        
        self.pen_thickness = 30
        self.mouse_x = None
        self.mouse_y = None
        self.is_drawing = False
        self.reset_frame()
    
    def run(self):
        cv2.namedWindow(constants.WINDOW)
        cv2.setMouseCallback(constants.WINDOW, self.on_mouse)
        
        while(1):
            if self.should_refresh:
                cv2.imshow(constants.WINDOW, self.frame)
                self.should_refresh = False

            if self.should_predict and self.frame.any(): # img.any(): skip if canvas is empty
                cv2.imshow(constants.PREDICTION_WINDOW, self.draw_prediction_window())
                self.should_predict = False

            key = cv2.waitKey(10)
            if key == 27 or key == ord('q'):
                break
            elif key == ord('c'):
                self.reset_frame()
                self.should_refresh = True
        cv2.destroyAllWindows()
        
    def predict(self):
        small = image_to_emnist_format(self.frame)
        x = small.reshape(1, 28, 28, 1)
        probs = self.model.predict(x, verbose=0)[0]
        top = np.argsort(probs)[::-1][:10] # sort, reverse, keep first 10
        return small, probs, top
    
    def draw_prediction_window(self):
        SIZE = constants.PREDICTION_WINDOW_SIZE # 28 * 10, so each model pixel = 10x10 screen pixels
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        small, probs, top = self.predict()
        # left side with image
        big = cv2.resize(small, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        big = (big * 255).astype(np.uint8)
        big = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)

        # right side with guesses
        panel = np.zeros((SIZE, 220, 3), np.uint8)
        row_h = SIZE // len(top) 
        for rank, i in enumerate(top):
            y = (rank + 1) * row_h - 8
            color = constants.BGR_GRAY
            if rank == 0: color = constants.BGR_GREEN

            cv2.putText(panel, f"{constants.RUSSIAN_LETTERS[i]} {probs[i]:6.1%}", (10, y), FONT, 0.6, color, 1, cv2.LINE_AA)
            bar_w = int(probs[i] * 100)              
            cv2.rectangle(panel, (110, y - 14), (110 + bar_w, y), color, -1) 
        return np.hstack([big, panel])
    
    def on_mouse(self, event, x, y, _flags, _params):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.mouse_x, self.mouse_y = x, y
            return
        
        elif event ==cv2.EVENT_MOUSEMOVE and self.is_drawing:
            cv2.line(self.frame, (self.mouse_x, self.mouse_y), (x, y), color=constants.BGR_WHITE, thickness=self.pen_thickness)
            self.mouse_x, self.mouse_y = x, y
            self.should_refresh = True
        
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.line(self.frame, (self.mouse_x, self.mouse_y), (x, y), color=constants.BGR_WHITE, thickness=self.pen_thickness)
            self.is_drawing = False
            self.should_refresh = True
            self.should_predict = True
            
    def reset_frame(self):
        self.frame = np.zeros((512,512,1), np.uint8)