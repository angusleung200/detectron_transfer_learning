from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
import pafy
import glob
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer



cfg = get_cfg()
cfg.merge_from_file('C:\\Users\\angus\\OneDrive\\桌面\\detectron\\output\\config.yaml')
cfg.MODEL.WEIGHTS = "C:\\Users\\angus\\OneDrive\\桌面\\detectron\\output\\model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)
print(predictor)
MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes = ['kangaroowallaby','kangaroo','wallaby']
url = 'https://youtu.be/fc-Lt6Hsgc0'
video = pafy.new(url)
best = video.getbest(preftype="mp4")
capture = cv2.VideoCapture(best.url)
while (True):
    grabbed, im = capture.read()
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('', v.get_image()[:, :, ::-1])
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
