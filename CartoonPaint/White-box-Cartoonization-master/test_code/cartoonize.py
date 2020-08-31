'''
Description: 
version: 
Author: HLLI8
Date: 2020-08-12 15:44:49
LastEditors: HLLI8
LastEditTime: 2020-08-31 14:46:07
'''
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


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720*h/w), 720
        else:
            h, w = 720, int(720*w/h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h//8)*8, (w//8)*8
    image = image[:h, :w, :]
    return image
def read_image(image_path):
    '''
    note: TODO:读图片显示 
    Author: HLLI8
    Date: 2020-08-17 16:21:03
    '''
    print("read_image" + str(image_path))
    image_path_t = image_path.replace('\\', '/')
    print("image_path_t"+ str(image_path_t))
    image = cv2.imread(image_path_t)
    image = cv2.resize(image, (720, 720))
    cv2.imshow("readimage", image)
    cv2.waitKey(0)
    cv.destroyAllWindows()
def out_image(out_image_path):
    '''
    note: TODO:经过网络处理后图片显示 
    Author: HLLI8
    Date: 2020-08-17 16:21:37
    '''    
    print("read_image" + str(out_image_path))
    out_image_path_t = out_image_path.replace('\\', '/')
    print("out_image_path_t"+str(out_image_path_t))
    image = cv2.imread(out_image_path_t)
    # image = cv2.resize(image, (720, 720))
    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv.destroyAllWindows()

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
        try:
            load_path = os.path.join(load_folder, name)
            save_path = os.path.join(save_folder, name)
            image = cv2.imread(load_path)
            image = resize_crop(image)
            t1 = threading.Thread(target=read_image, args=(load_path, ))
            t1.start()
            batch_image = image.astype(np.float32)/127.5 - 1
            batch_image = np.expand_dims(batch_image, axis=0)
            output = sess.run(final_out, feed_dict={input_photo: batch_image})
            output = (np.squeeze(output)+1)*127.5
            output = np.clip(output, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, output)
            t2 = threading.Thread(target=out_image, args=(save_path, ))
            t2.start()
        except:
            print('cartoonize {} failed'.format(load_path))
    sess.close()


    

if __name__ == '__main__':
    model_path = 'E:/PythonWorkSpace/DeepLearningWithOpenCV/CartoonPaint/White-box-Cartoonization-master/test_code/saved_models'
    load_folder = 'E:/PythonWorkSpace/DeepLearningWithOpenCV/CartoonPaint/White-box-Cartoonization-master/test_code/demo_image'
    save_folder = 'E:/PythonWorkSpace/DeepLearningWithOpenCV/CartoonPaint/White-box-Cartoonization-master/test_code/demo_image_output'
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    cartoonize(load_folder, save_folder, model_path)
    

    