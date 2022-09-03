from torch import optim
from darknet import *
from load_data import MaxProbExtractor_yolov2, MaxProbExtractor_yolov5, MeanProbExtractor_yolov5, \
    MeanProbExtractor_yolov2
from mmdet.apis.inference import InferenceDetector
from models.common import DetectMultiBackend
from models_yolov3.common import DetectMultiBackend_yolov3
from utils_yolov5.torch_utils import select_device, time_sync
import os
from mmdet.apis import (async_inference_detector, InferenceDetector,
                        init_detector, show_result_pyplot)


class BaseConfig(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        # self.img_dir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages"
        self.img_dir = "testing/plane_random_loc/clean"
        # self.lab_dir = "datasets/RSOD-Dataset/aircraft/Annotation/JPEGImages/labels-yolo"
        self.lab_dir = "testing/yolov5l_center_150_1024_yolov5l/clean/labels-yolo"
        self.printfile = "non_printability/30values.txt"
        self.patch_size = 50

        self.start_learning_rate = 0.03

        self.patch_name = 'base'

        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.max_tv = 0

        self.batch_size = 1

        self.loss_target = lambda obj, cls: obj * cls  # self.loss_target(obj, cls) return obj * cls


class Experiment1(BaseConfig):
    """
    Model that uses a maximum total variation, tv cannot go below this point.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment1'
        self.max_tv = 0.165


class Experiment2HighRes(Experiment1):
    """
    Higher res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 400
        self.patch_name = 'Exp2HighRes'


class Experiment3LowRes(Experiment1):
    """
    Lower res
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.max_tv = 0.165
        self.patch_size = 100
        self.patch_name = "Exp3LowRes"


class Experiment4ClassOnly(Experiment1):
    """
    Only minimise class score.
    """

    def __init__(self):
        """
        Change stuff...
        """
        super().__init__()

        self.patch_name = 'Experiment4ClassOnly'
        self.loss_target = lambda obj, cls: cls


class Experiment1Desktop(Experiment1):
    """
    """

    def __init__(self):
        """
        Change batch size.
        """
        super().__init__()

        self.batch_size = 8
        self.patch_size = 400


class ReproducePaperObj(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.batch_size = 1
        self.patch_size = 150

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # yolov5
        # 15,exp:n, 16,19:s, 9:m, 17,24:l, 18,26:x
        self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5n/weights/best.pt'
        self.device = select_device('')
        self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS


class yolov2(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.cfgfile = "cfg/yolo-dota.cfg"
        self.weightfile = "weights/yolo-dota.cfg_450000.weights"

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weightfile)
        self.model = self.model.eval().cuda()

        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image

        # self.prob_extractor = MaxProbExtractor_yolov2(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov2(0, 15, self.loss_target, self.conf_thres, self.iou_thres, self.max_det, self.model)


class yolov3(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # 2080
        # self.weights_yolov3 = '/home/mnt/ljw305/yolov3/runs/train/exp5/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov3/data/DOTA1_0.yaml'
        # 3080
        self.weights_yolov3 = "/home/ljw/yolov3-master/runs/train/exp5/weights/best.pt"
        self.data = '/home/ljw/yolov3-master/data/DOTA1_0.yaml'

        self.device = select_device('')
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = DetectMultiBackend_yolov3(self.weights_yolov3,
                                               device=self.device,
                                               dnn=False, ).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres, self.max_det)


class yolov5n(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5n/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'

        # 3080
        self.weights_yolov5 = '/home/ljw/yolov5/runs/train/yolov5n/weights/best.pt'
        self.data = '/home/ljw/yolov5/data/DOTA1_0.yaml'

        self.device = select_device('')
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres, self.max_det)

class yolov5s(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5s/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'

        # 3080
        self.weights_yolov5 = '/home/ljw/yolov5/runs/train/yolov5s/weights/best.pt'
        self.data = '/home/ljw/yolov5/data/DOTA1_0.yaml'

        self.device = select_device('')
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres, self.max_det)

class yolov5m(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5m/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'

        # 3080
        self.weights_yolov5 = '/home/ljw/yolov5/runs/train/yolov5m/weights/best.pt'
        self.data = '/home/ljw/yolov5/data/DOTA1_0.yaml'

        self.device = select_device('')
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres, self.max_det)

class yolov5l(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5l/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'

        # 3080
        self.weights_yolov5 = '/home/ljw/yolov5/runs/train/yolov5l/weights/best.pt'
        self.data = '/home/ljw/yolov5/data/DOTA1_0.yaml'

        self.device = select_device('')
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres, self.max_det)

class yolov5x(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        # 2080
        # self.weights_yolov5 = '/home/mnt/ljw305/yolov5/runs/train/yolov5x/weights/best.pt'
        # self.data = '/home/mnt/ljw305/yolov5/data/DOTA1_0.yaml'

        # 3080
        self.weights_yolov5 = '/home/ljw/yolov5/runs/train/yolov5x/weights/best.pt'
        self.data = '/home/ljw/yolov5/data/DOTA1_0.yaml'

        self.device = select_device('')
        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = DetectMultiBackend(self.weights_yolov5,
                                        device=self.device,
                                        dnn=False,
                                        data=self.data).eval()
        # self.prob_extractor = MaxProbExtractor_yolov5(0, 15, self.loss_target)
        self.prob_extractor = MeanProbExtractor_yolov5(0, 15, self.loss_target, self.conf_thres, self.iou_thres, self.max_det)

class faster_rcnn(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/faster-rcnn/epoch_12.pth'
        # 3080
        self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/faster-rcnn/epoch_12.pth'

        self.device = 'cuda:0'

        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)

        self.InferenceDetector = InferenceDetector()


class ssd(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.config_file = 'configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py'

        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/ssd/epoch_120.pth'
        # # 3080
        self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/ssd/epoch_120.pth'

        self.device = 'cuda:0'

        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)

        self.InferenceDetector = InferenceDetector()


class swin(BaseConfig):
    """
    Reproduce the results from the paper: Generate a patch that minimises object score.
    """

    def __init__(self):
        super().__init__()

        self.patch_name = 'ObjectOnlyPaper'
        self.max_tv = 0.165

        self.loss_target = lambda obj, cls: obj

        self.config_file = 'configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco.py'

        # 2080
        # self.checkpoint_file = '/home/mnt/ljw305/mmdetection-master/work_dirs/swin/epoch_24.pth'
        # 3080
        self.checkpoint_file = '/home/ljw/mmdetection-master/work_dirs/swin/epoch_24.pth'

        self.device = 'cuda:0'

        self.img_size = 1024
        self.imgsz = (1024, 1024)
        self.conf_thres = 0.4  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS

        self.model = init_detector(self.config_file, self.checkpoint_file, device=self.device)

        self.InferenceDetector = InferenceDetector()


patch_configs = {
    "base": BaseConfig,
    "exp1": Experiment1,
    "exp1_des": Experiment1Desktop,
    "exp2_high_res": Experiment2HighRes,
    "exp3_low_res": Experiment3LowRes,
    "exp4_class_only": Experiment4ClassOnly,
    "paper_obj": ReproducePaperObj,
    "yolov2": yolov2,
    "yolov3": yolov3,
    "yolov5n": yolov5n,
    "yolov5s": yolov5s,
    "yolov5m": yolov5m,
    "yolov5l": yolov5l,
    "yolov5x": yolov5x,
    "faster-rcnn": faster_rcnn,
    "swin": swin,
    "ssd": ssd,
}
