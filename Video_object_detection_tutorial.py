import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
 # 기존 학습되어있는 모델로 Parameter 값을 갖고 영상에서 특정 모델로 부터 학습된 weight로 video영상으로 detection 하는 코드
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from datetime import datetime
import time
import math

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
#print(os.listdir())
filename = os.listdir()
#if 'ssd_mobilenet_v1_coco_11_06_2017' not in filename:
MODEL_NAME = 'inference_graph'
#MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb' #

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('training', 'labelmap.pbtxt') # label(CLASSES)

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
  #file_name = os.path.basename(file.name)
  #if 'frozen_inference_graph.pb' in file_name:
    #tar_file.extract(file, os.getcwd())

#NumPy의 다차원배열 ndarray 클래스
#tf.Graph 클래스로 명시적으로 그래프를 생성할 수도 있다. 사용자가 생성한 그래프를 사용하려면 as_default 메서드로 with 블럭을 만들어 사용한다.***
#TensorFlow에서 계산의 기본은 Graph 객체
#모든 텐서 계산은 해당하는 텐서를 포함하는 그래프를 세션 객체에 전달하여 원격 실행한 후에야 값을 볼 수 있다.
#프로토콜 버퍼 툴은 이 텍스트 파일을 파싱하고 그래프 정의를 로딩, 저장 및 조작하는 코드를 생성
detection_graph = tf.Graph()
with detection_graph.as_default(): #사용자가 생성한 그래프 사용(/frozen_inference_graph.pb)
  od_graph_def = tf.GraphDef()# graph를 생성한 후에는 graphdef 객체를 반환하는 as_graph_def를 호출함으로써 이를 저장
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid: #tf.gfile.GFile는 open()함수와 유사 / 텐서플로 입출력함수 , 'Read Binary'
    #serialized_graph = fid.read() #Custom Graph를 읽음 -> Serialized_graph 변수에
    od_graph_def.ParseFromString(fid.read())
    tf.import_graph_def(od_graph_def, name='') #첫인자:custom or default graph, 두번째인자:name(graph_def의 이름에 붙일 접두사.)
#그래프를 graph_def에서 현재 그래프로 가져오기. == 학습된 data를 가져옴

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
#label_map class정보 가져오기(배열)

'''
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3).astype(np.uint8))

PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image().jpg'.format(i)) for i in range(1, 3)]
'''

IMAGE_SIZE = 12, 8

def run_inference_for_single_image(image, graph):
    if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        print(len(detection_boxes, len(detection_masks)))
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.8), tf.uint8) # (a, b)  a>b True, False값 반환
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

import cv2 #영상, 이미지 data를 이용하는 Library
capture = cv2.VideoCapture("road_960_512.mp4")

size = (
    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
Video_w = (capture.get(cv2.CAP_PROP_FRAME_WIDTH))
Video_h = (capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)
print("fps :", fps)
length = frame/fps

print("Width :", Video_w, "Height :", Video_h)

Video_w_20 = round(Video_w * 0.2)#반올림 함수 round
Video_w_80 = round(Video_w - Video_w_20)
Video_h_35 = round(Video_h * 0.35)

print(Video_w_20, Video_w_80, Video_h_35)
#Video_w_20 : 화면 상 좌 20% 지점 / Video_w_80 : 화면 상 우 80% 지점

#codec = cv2.VideoWriter_fourcc(*'DIVX')
#output = cv2.VideoWriter('videofile_masked_road_20%35%_4x_50_inc.avi', codec, 30.0, size)

flag = 0
masking = 0 # 인코딩(Boxing) 처리 성능 향상을 위해서 한프레임씩 건너서 Boxing(Object Detecting) -> 속도 향상
print("Start masking")
now = datetime.now()
print("Start at :", now)
start = round(time.time())
#

try:
    with detection_graph.as_default():
        with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                  'num_detections', 'detection_boxes', 'detection_scores',
                  'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                      tensor_name)

                while True:
                    ret, image_np = capture.read() # Video에서 받은 capture 1frame씩 읽기 (ret : 받았는지 정보, image_np(2번째 인자) : 이미지 frame size(,))

                    if ret and masking == 0:
                        masking = masking + 1

                        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                        image_np_expanded = np.expand_dims(image_np, axis=0)#axis:0 : Row(행)-->   / axis:1 : Column방향
                        # Actual detection.
                        output_dict = run_inference_for_single_image(image_np, detection_graph)
                        # Visualization of the results of a detection.
                        vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                            use_normalized_coordinates=True,
                            line_thickness=8)
                        #output.write(image_np)
                        cv2.imshow('frame', image_np)  # 원본 영상에 Masking이 입혀진 영상 보여주기 함수
                        #cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    elif ret and masking > 0:
                        masking = masking + 1
                        if masking == 4:  # 몇 프레임 당 Compute할것인지
                            masking = 0
                        if cv2.waitKey(1) & 0xFF == ord('q'):  # waitkey & 0xFF = 1111 1111 == 'q'
                            break
                        #output.write(image_np)  # Model forward Compute를 거치지 않고 바로 출력
                        cv2.imshow('Drive', image_np)
                    elif cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    else:
                        break
except Exception as e:
    print(e)
    capture.release()

now = datetime.now()
print("End at :", now)

end = round(time.time())
taken_time = end - start
minute = math.floor(taken_time/60)
sec = taken_time%60
print("taken_time :", minute, ":", sec)

rate = length/taken_time
print("encoding rate :", rate, ": 1", "1보다 커야 실시간O")

capture.release()
#output.release()
cv2.destroyAllWindows()