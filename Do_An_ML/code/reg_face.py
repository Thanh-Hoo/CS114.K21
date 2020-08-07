from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import logging
logging.getLogger('tensorflow').disabled = True
import facenet
import pickle
import detect_face
import numpy as np
import cv2
MINSIZE = 20
THRESHOLD = [0.6, 0.7, 0.7]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
CLASSIFIER_PATH = 'code/model/facemodels.pkl'
FACENET_MODEL_PATH = 'code/model/20180402-114759.pb'

# Load The Custom Classifier
with open(CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)
print("Custom Classifier, Successfully loaded")

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6) #đặt giới hạn cứng trên đối với bộ nhớ GPU được phép dùng
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) #tạo môi trường cho TF thực thi việc tính toán ngay trên GPU

# Load the model
print('Loading feature extraction model')
facenet.load_model(FACENET_MODEL_PATH)

# Get input and output tensors để lấy đặc trưng từ ảnh
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
pnet, rnet, onet = detect_face.create_mtcnn(sess, "code/align")

def Recognizing():
    with tf.Graph().as_default():
        with sess.as_default():
            cap = cv2.VideoCapture(0)
            try:
                while (True):
                    __, frame = cap.read()
                
                    bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                    faces_found = bounding_boxes.shape[0] #Number of faces found
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            if (bb[i][3]-bb[i][1])/frame.shape[0]>0.1: #Loại bỏ những khuôn mặt quá nhỏ trong hình
                                
                                cropp = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                                scaled = cv2.resize(cropp, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                                    interpolation=cv2.INTER_CUBIC) #resize lại ảnh theo chuẩn đầu vào của model

                                scaled = facenet.prewhiten(scaled)
                                
                                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                                
                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                
                                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                                
                                predictions = model.predict_proba(emb_array)
                                print("predictions: ",predictions)
                                #vị trí của labels giống nhất với input
                                best_class_indices = np.argmax(predictions, axis=1)
                                print("best_class_indices: ",best_class_indices)
                                #xác suất của lớp tốt nhất   
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                                        
                                #label của class đó ứng với tên của người đó
                                dict_name = model.predict(emb_array)
                                name_r = dict_name[0]
                                

                                #Xét ngưỡng để xác định danh tính khuôn mặt 
                                if best_class_probabilities > 0.70:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    cv2.putText(frame, name_r, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                                else:
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                                    text_x = bb[i][0]
                                    text_y = bb[i][3] + 20

                                    name = "Unknown"
                                    cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                        

                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except:
                pass
            
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    print("class_names: ", class_names)
    print('type_class_names: ', type(class_names))
    Recognizing()
