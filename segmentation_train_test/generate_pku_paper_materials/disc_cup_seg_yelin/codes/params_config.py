class param_config(object):
	def __init__(self):
		## models path
		self._disc_detector_weight = '../models/disc_detector.caffemodel'
		self._disc_detector_model = '../models/deploy_disc_detector.prototxt'
		self._disc_det_mean = [104.01, 116.67, 122.68] #  B: 104.01 , G: 116.67, R: 122.68

		self._disc_seg_weight = '../models/disc_seg.caffemodel'
		self._disc_seg_model = '../models/deploy_disc_seg.prototxt'

		self._cup_seg_weight = '../models/cup_seg.caffemodel'
		self._cup_seg_model = '../models/deploy_cup_seg.prototxt'

		self._gpu_id_0 = 0
		self._gpu_id_1 = 1

	def disc_det_proto(self):
		return self._disc_detector_model

	def disc_det_weight(self):
		return self._disc_detector_weight

	def disc_det_mean(self):
        	return self._disc_det_mean

	def disc_seg_proto(self):
		return self._disc_seg_model

	def disc_seg_weight(self):
		return self._disc_seg_weight

	def cup_seg_proto(self):
		return self._cup_seg_model

	def cup_seg_weight(self):
		return self._cup_seg_weight

	def get_gpu_id_0(self):
        	return self._gpu_id_0
	def get_gpu_id_1(self):
        	return self._gpu_id_1

params = param_config()

