#Phần này để Train và lưu model

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pickle
import os
from sklearn.svm import SVC
import pandas as pd


def main():

            df = pd.read_csv('features.csv')          
            labels = df.iloc[:,-1].values #Lấy nhãn tương ứng
            emb_array = df.iloc[:,0:-1].values#Lấy đặc trưng tương ứng

            classifier_file = os.path.expanduser("code/model/facemodels.pkl")# tạo đường dẫn model
            
            sav = pd.read_csv('labels.csv', header=None)
            model = SVC(kernel='linear', probability=True)# cài đặt đối số cho SVM
            model.fit(emb_array, labels)#tiến hành train model SVM


            # Create a list of class names
            class_names  = sav.iloc[:,0].values
                
            print("class_names: ", class_names)
            
            with open(classifier_file, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('DONE')             

main() 

