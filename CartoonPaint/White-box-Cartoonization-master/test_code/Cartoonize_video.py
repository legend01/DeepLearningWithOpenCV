import sys
sys.path.append ("D:/ProgramFile/Anaconda/Lib/site-packages") 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #只显示warning和Error

import os
import cv2
import numpy as np
import tensorflow as tf 
import network
import guided_filter
from tqdm import tqdm

import threading

def original_image(image):
    cv2.imshow("Original", image)
    cv2.waitKey(0)
    cv.destroyAllWindows()

def output_image(image):
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv.destroyAllWindows()

def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    print("[INFO]resize_crop:h: "+str(h)+" w: "+str(w)+" c: "+str(c))
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]

    t1 = threading.Thread(target=original_image, args=(image, ))
    t1.start()
    # cv2.imshow("Original", image)
    # cv2.waitKey(10)

    return image

def cartoonize_image(image, final_out, sess, input_photo):
    image = resize_crop(image)
    batch_image = image.astype(np.float32)/127.5 - 1
    batch_image = np.expand_dims(batch_image, axis=0)
    output = sess.run(final_out, feed_dict={input_photo: batch_image})
    output = (np.squeeze(output)+1)*127.5
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def video_analysis(video_path, final_out, sess, input_photo, output_path = ""):
    Count_frame = 0
    print("[INFO]Video_analysis:"+str(video_path)+str("......."))
    video_path_T = video_path.replace('\\', '/')
    print("[INFO]Video_analysis transform : "+str(video_path_T)+str("......."))
    
    print("[INFO]output_path:"+str(output_path)+str("......."))
    output_path = output_path.replace('\\', '/')
    print("[INFO]output_path transform : "+str(output_path)+str("......."))

    vid = cv2.VideoCapture(video_path_T)  # 使用OpenCV打开USB相机,读取视频
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC)) # 编码方式
    video_fps       = vid.get(cv2.CAP_PROP_FPS) # 读取视频FPS值
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), #读取视频大小
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    print("video_FourCC: "+str(video_FourCC)+"video_fps: "+str(video_fps)+"video_size: "+str(video_size))
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, video_size)  #创建写入对象
    success, image = vid.read()  # 读取视频
    while success:
        output = cartoonize_image(image, final_out, sess, input_photo)  # 开始对每帧进行检测
        '''****************************************************************'''
        # image = resize_crop(image)
        # batch_image = image.astype(np.float32)/127.5 - 1
        # batch_image = np.expand_dims(batch_image, axis=0)
        # output = sess.run(final_out, feed_dict={input_photo: batch_image})
        # output = (np.squeeze(output)+1)*127.5
        # output = np.clip(output, 0, 255).astype(np.uint8)


        t2 = threading.Thread(target=output_image, args=(output, ))
        t2.start()
        # cv2.imshow("output", output)
        # cv2.waitKey(10)
        '''****************************************************************'''
        result = np.asarray(output)
        if isOutput:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        success, image = vid.read()
        Count_frame += 1
        print("[INFO] Ongoing programing "+str(Count_frame)+".....")

def cartoonize(load_folder, save_folder, model_path):
    input_photo = tf.placeholder(tf.float32, [1, None, None, 3])
    network_out = network.unet_generator(input_photo)
    final_out = guided_filter.guided_filter(input_photo, network_out, r=1, eps=5e-3)

    all_vars = tf.trainable_variables()
    gene_vars = [var for var in all_vars if 'generator' in var.name]
    saver = tf.train.Saver(var_list=gene_vars)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path))
    name_list = os.listdir(load_folder)
    for name in tqdm(name_list):
        print("[INFO] for circle name " + str(name) + "....")
        # try:
        load_path = os.path.join(load_folder, name)
        save_path = os.path.join(save_folder, name)

        video_analysis(load_path, final_out, sess, input_photo, save_path)
        # except:
        #     print('cartoonize {} failed'.format(load_path))
    sess.close()


    

if __name__ == '__main__':
    model_path = 'E:/PythonWorkSpace/DeepLearningWithOpenCV/CartoonPaint/White-box-Cartoonization-master/test_code/saved_models'
    load_folder = 'E:/PythonWorkSpace/DeepLearningWithOpenCV/CartoonPaint/White-box-Cartoonization-master/test_code/demo_video'
    save_folder = 'E:/PythonWorkSpace/DeepLearningWithOpenCV/CartoonPaint/White-box-Cartoonization-master/test_code/demo_video_output'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
    

    