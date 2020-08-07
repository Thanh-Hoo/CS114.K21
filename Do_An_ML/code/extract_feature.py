#Phần này là lấy data đầu vào

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import facenet
import os
import detect_face
import numpy as np
import cv2
from sklearn.svm import SVC
import pandas as pd

MINSIZE = 20 #ngưỡng thấp nhất khi resize
THRESHOLD = [0.6, 0.7, 0.7]#thiết lập ngưỡng của ảnh, mức độ confidence để nhận mặt. Array ba giá trị cho ba mạng.
FACTOR = 0.7#thiết lập độ tương phản
INPUT_IMAGE_SIZE = 160
FACENET_MODEL_PATH = 'code/model/20180402-114759.pb'#đường dẫn model

tf.Graph().as_default() #Thêm note vào graph default


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6) #đặt giới hạn cứng trên đối với bộ nhớ GPU được phép dùng
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) #tạo môi trường cho TF thực thi và tính toán 


#Nhập vào tên nhân viên và đưa tên đó vào file labels.csv
name=input('Nhập tên nhân viên:')

#lưu tên nhân viên vào file label
if not os.path.isfile("labels.csv"):
    with open("labels.csv", "a") as f:
        f.write('0,')
        f.write(name)
        f.write('\n')
else:
    df_labels = pd.read_csv("labels.csv")
    my_ids = df_labels.shape[0]
    my_ids += 1
    my_ids = [my_ids]
    sav=pd.concat([pd.DataFrame(my_ids),pd.Series(name)],axis=1)
    sav.to_csv("labels.csv", mode='a', header=None, index=False)
            
# kiểm tra và tạo folder chứ hình ảnh mặt cắt khuôn mặt và lưu vào đó
folder = "Dataset/" + str(name) +"/"
if not os.path.exists(folder):
    os.makedirs(folder)

# Load the model facenet để detect khuôn mặt và lấy đặc trưng của khuôn mặt đó
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors để lấy đặc trưng từ ảnh 
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0") 
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") #Quản lý graph 
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
pnet, rnet, onet = detect_face.create_mtcnn(sess, "code/align")

# Đầu vào hình ảnh được resize thành nhiều kích thước tạo thành một Image Pyramid. Sau đó pyramid sẽ được đưa vào P-Net

def Extract_feature():
    #Time start
    MINSIZE = 20 #Size ảnh nhỏ nhất
    THRESHOLD = [0.6, 0.7, 0.7] #thiết lập ngưỡng của ảnh, mức độ confidence để nhận mặt. Array ba giá trị cho ba mạng.
    FACTOR = 0.7 #đơn vị scale ảnh trên pyramid
    INPUT_IMAGE_SIZE = 160 #cỡ ảnh đầu vào yêu cầu
    
    if not os.path.exists("Dataset"):
        os.mkdir("Dataset")
    
    cap = cv2.VideoCapture(0)
    cnt = 0
    while (True):
        __, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)

        bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]
        print("faces_found: ", faces_found)
    
        if bounding_boxes != []:
            flag = 1
            # Lấy ra toạ độ của X_max, y_max, X_min, y_min trong "bounding_boxes"
            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2) #Vẽ khung theo các toạn độ của bounding_boxes đã có

                
                if (bb[i][3]-bb[i][1])/frame.shape[0]>0.0:
                    cropp = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :] # crop hình ảnh theo toạ độ X_max, y_max, X_min, y_min trong "bounding_boxes"
                    scaled_out = cv2.resize(cropp, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),interpolation=cv2.INTER_CUBIC) # resize ảnh frame vừa mới crop
                    
                    #dùng facenet trích xuất đặc trưng khuôn mặt
                    scaled = facenet.prewhiten(scaled_out)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    
                    #print("emb_array.shape: ", emb_array.shape)
                    #Lưu ảnh vào folder Dataset
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    cv2.imwrite(folder + str(cnt) + '.jpg', scaled_out)
                    #Lưu feature vừa trích xuất được ở trên vào file feature.csv
                    emb_array = np.append(emb_array, name)
                    my_features = np.array(emb_array)
                    my_features = my_features.reshape(-1, my_features.shape[0])
                    df = pd.DataFrame(my_features)
                    df.to_csv("features.csv", mode='a', header=None, index=False)
                    cnt += 1

        cv2.imshow('Face Recognition', frame)#Show hình ảnh camera dưới dạng cửa sổ với tên là Face Recognition
        if cv2.waitKey(10) & 0xFF == ord('q'):#Lệnh thoát khỏi cửa sổ
            break
        elif cnt>100:
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    Extract_feature()
