import streamlit as st
import onnxruntime as rt
from PIL import Image
import os
import numpy as np
import cv2

target_size = 640.

def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map

classes = [
    'Red',
    'Yellow',
    'Green',
    'Off'
]


def model_process(img_path, session):
    inputs = {}
    input_names = [input.name for input in session.get_inputs()]
    output_names = [output.name for output in session.get_outputs()]
    img = cv2.imread(img_path)
    origin_shape = img.shape[:2]
    im_scale_x = im_scale_y = 1.0
    scale_factor = np.array([[im_scale_y, im_scale_x]]).astype('float32')
    im = cv2.resize(img,(int(target_size), int(target_size)))
    im = im / 255.0
    mean = [0.485, 0.456, 0.406]
    std =[0.229, 0.224, 0.225]
    im = (im - mean) / std
    im = im[:, :, ::-1]
    im = np.expand_dims(np.transpose(im, (2, 0, 1)), axis=0)
    inputs['im_shape'] =  np.array([origin_shape]).astype('float32')
    inputs['scale_factor'] = scale_factor
    inputs['image'] = im.astype('float32') 
    np_boxes = session.run(output_names, inputs)[0]
    expect_boxes = (np_boxes[:, 1] > 0.2) & (np_boxes[:, 0] > -1)
    np_boxes = np_boxes[expect_boxes, :]

    color_list = get_color_map_list(4)
    clsid2color = {}

    for i in range(np_boxes.shape[0]):
        classid, conf = int(np_boxes[i, 0]), np_boxes[i, 1]
        xmin, ymin, xmax, ymax = int(np_boxes[i, 2]), int(np_boxes[
            i, 3]), int(np_boxes[i, 4]), int(np_boxes[i, 5])

        if classid not in clsid2color:
            clsid2color[classid] = color_list[classid]
        color = tuple(clsid2color[classid])
        cv2.rectangle(
            img, (xmin, ymin), (xmax, ymax), color, thickness=2)
        print(classes[classid] + ': ' + str(round(conf, 3)))
        cv2.putText(
            img,
            classes[classid] + ':' + str(round(conf, 3)), (xmin,
                                                                ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0),
            thickness=2)
    cv2.imwrite('images/pred.png', img)




if __name__ == '__main__':
    st.title("【AI 达人训练营】 交通灯检测")
    st.image(['./images/tag.jpg'])
    st.write('左侧上传文件')
    # 模型位置
    sess = rt.InferenceSession("weights/ppyolov2_infer_quant_dynamic.onnx")
    source = None
    uploaded_file = st.sidebar.file_uploader("上传图片", type=['png', 'jpeg', 'jpg', 'bmp'])

    if uploaded_file is not None:
        is_valid = True

        with st.spinner(text='资源加载中...'):
            st.sidebar.image(uploaded_file)
            picture = Image.open(uploaded_file)
            picture = picture.save(f'images/{uploaded_file.name}')
            source = f'images/{uploaded_file.name}'
    else:
        is_valid = False

    if is_valid:
        image = model_process(source, sess)
        with st.spinner(text='Preparing Images'):
            st.image('images/pred.png')
