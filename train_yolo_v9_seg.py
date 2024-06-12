from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits PyDataProcess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate algorithm object
        from train_yolo_v9_seg.train_yolo_v9_seg_process import TrainYoloV9SegFactory
        return TrainYoloV9SegFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from train_yolo_v9_seg.train_yolo_v9_seg_widget import TrainYoloV9SegWidgetFactory
        return TrainYoloV9SegWidgetFactory()
