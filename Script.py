# %%
"""
    image should be in the same directory of the notebook
"""
def deetctAndCropt(image_name):
    
    import os   
    
    if not (os.path.abspath(image_name)):
        print("image doesnt exist")
        return;
    
    import tensorflow as tf
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as viz_utils
    from object_detection.builders import model_builder
    from object_detection.utils import config_util
    import cv2 
    import numpy as np
    from matplotlib import pyplot as plt
 
    IMAGE_PATH = image_name
    CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
    PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    LABEL_MAP_NAME = 'label_map.pbtxt'
    paths = {
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
    }
    files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
    }
    
    labels = [{'name':'licence', 'id':1}]


 
    import subprocess
    import sys 
    subprocess.check_call([sys.executable,"-m","pip","install","wget"])

    # Install Tensorflow Object Detection 
    import sys

    # For illustrative purposes.
    package = 'object_detection'

    if package in sys.modules:
        print(f"{package!r} already in sys.modules")
    else:
        print(f"can't find the {package!r} module")
        url="https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
        wget.download(url)
        from subprocess import check_output
        check_output("move protoc-3.15.6-win64.zip {}".format(paths['PROTOC_PATH']), shell=True)
        check_output("cd {} && tar -xf protoc-3.15.6-win64.zip".format(paths['PROTOC_PATH']), shell=True)
        os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(paths['PROTOC_PATH'], 'bin'))   
        check_output("cd Tensorflow/models/research && protoc object_detection/protos/*.proto --python_out=. && copy object_detection\\packages\\tf2\\setup.py setup.py && python setup.py build && python setup.py install", shell=True)
        check_output("cd Tensorflow/models/research/slim && pip install -e .", shell=True)
        import IPython
        IPython.Application.instance().kernel.do_shutdown(True) #automatically restarts kernel
    
    import object_detection
    

            
    if not os.path.exists(paths['IMAGE_PATH']):
        if os.name == 'posix':
            check_output("mkdir -p {}".format(paths['IMAGE_PATH']), True)
            check_output("mkdir -p {}".format(paths['ANNOTATION_PATH']), True)
        if os.name == 'nt':
            check_output("mkdir  {}".format(paths['IMAGE_PATH']), True)
            check_output("mkdir  {}".format(paths['ANNOTATION_PATH']), True)
            
    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
    detection_model = model_builder.build(model_config=configs['model'], is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-11')).expect_partial()
    print(paths['CHECKPOINT_PATH'])
 
    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections
    
    category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])
    
    
    SAVE_DIR_PATH = os.path.join(paths['IMAGE_PATH'], 'save')
    if not os.path.exists(SAVE_DIR_PATH):
               check_output("mkdir  {}".format(paths SAVE_DIR_PATH), True)
    SAVE_IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'save',image_name.split("\\")[-1])
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)
    #get image size
    # get dimensions of image
    dimensions = img.shape

    # height, width, number of channels in image
    im_height = img.shape[0]
    im_width = img.shape[1]
    channels = img.shape[2]

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)


    boxes = detections['detection_boxes']
    ymin = int((boxes[0][0]*im_height))
    xmin = int((boxes[0][1]*im_width))
    ymax = int((boxes[0][2]*im_height))
    xmax = int((boxes[0][3]*im_width))

    print ("xmin: {} ".format(xmin),"ymin: {}".format(ymin),"xmax: {}".format(xmax),"ymax: {}".format(ymax))

    cropped_img=tf.image.crop_to_bounding_box(
                image=img, 
                offset_height=ymin, 
                offset_width=xmin, 
                target_height=ymax-ymin, 
                target_width=xmax-xmin
            )

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()
    image = np.array(cropped_img)

    print(image.shape)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    cv2.imwrite(SAVE_IMAGE_PATH, image)

    
deetctAndCropt(r"C:\Users\Administrator\Pictures\Camera Roll\1.jpeg") 



