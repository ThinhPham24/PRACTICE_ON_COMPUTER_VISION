import cv2
class ProcForOrchid():
    def __init__(self):
        # self.img_L = cv2.imread(path_ImgL)
        # self.img_L = path_ImgL
        print('run_crop')
    def crop_for_DL(self,path_ImgL):
        frame = path_ImgL
        # factor = 0.6
        # frame_shape = np.array(frame.shape[:2]) # y,x
        y, x, _ = (frame.shape)
        top = int(y * 0.25)
        left = int(x * 0.25)
        self.top = top
        self.left = left
        # print(left, top) # 1036 819
        bot = int(y * 0.85)
        right = int(x * 0.85)
        crop_img = frame[top:bot, left:right]
        # print(crop_img.shape) ## y = 921, x = 1167
        self.crop_img_shape = crop_img.shape
        crop_img = cv2.resize(crop_img, (640, 480))
        return crop_img

    def convert_arrow2orinal(self, path_name):
        '''

        :param path_name: arrow.txt
        :return: [[x1,y1,x2,y2],[x1,y1,x2,y2],[x1,y1,x2,y2]]
        '''
        left, top = 1036, 819 # 1036 819
        # image size should be y = 921, x = 1167
        factor_x = self.crop_img_shape[1] / 640
        factor_y = self.crop_img_shape[0] / 480
        with open(path_name) as f:
            lines = f.readlines()
            correct_line = []
            for num_line, line in enumerate(lines):
                # each_arrow_mask = np.zeros((2048, 2592), np.uint8)
                # print(line)
                line = line.split("\n") # avoid \n
                # print("line", line)
                ele = line[0].split(" ")
                # print("ele", ele)
                # for el in ele:
                # print(el)
                arrow = [self.left + int(int(ele[0]) * factor_x), self.top + int(int(ele[1]) * factor_y),
                self.left + int(int(ele[2]) * factor_x), self.top + int(int(ele[3]) * factor_y)]
                # print("arrow", arrow)
                correct_line.append(arrow)
        return correct_line