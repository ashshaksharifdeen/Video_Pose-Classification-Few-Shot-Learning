from __future__ import print_function
import argparse
import csv
import glob
import os
import sys
import random
import numpy as np
from tqdm import tqdm
from tqdm._tqdm import trange
from PIL import Image
import tensorflow as tf
import cv2
import time
import itertools

class DataGenerator:
      
    def __init__(self, args=None):
        
        '''
        :param mode: train or test
        :param n_way: a train task contains images from different N classes
        :param k_shot: k images used for meta-train
        :param k_query: k images used for meta-test
        :param meta_batchsz: the number of tasks in a batch
        :param total_batch_num: the number of batches
        '''

        if args is not None:
            
            self.mode = args.mode
            self.meta_batchsz = args.meta_batchsz
            self.n_way = args.n_way
            self.spt_num = args.k_shot
            self.qry_num = args.k_query
            self.dim_output = self.n_way

        self.img_size = 84
        self.img_channel = 3
        META_TRAIN_DIR = 'CAUCAFall/train'
        META_VAL_DIR = 'CAUCAFall/test'
        # Set sample folders
        self.metatrain_folders = [
                                    os.path.join(META_TRAIN_DIR, label, sublabel)\
                                            for label in os.listdir(META_TRAIN_DIR)\
                                            for sublabel in os.listdir((os.path.join(META_TRAIN_DIR, label)))\
                                                if os.path.isdir(os.path.join(META_TRAIN_DIR, label, sublabel))]
        
        self.metaval_folders = [
                                    os.path.join(META_VAL_DIR, label, sublabel)\
                                            for label in os.listdir(META_VAL_DIR)\
                                            for sublabel in os.listdir((os.path.join(META_VAL_DIR, label)))\
                                                if os.path.isdir(os.path.join(META_VAL_DIR, label, sublabel))]    
        
    def print_label_map(self):          
                print ('[TEST] Label map of current Batch')
                
                if len(self.label_map) > 0:
                        for i, task in enumerate(self.label_map):
                            print ('========= Task {} =========='.format(i+1))
                            for i, ref in enumerate(task):
                                path = ref[0]
                                label = path.split('/')[-1]
                                print ('map {} --> {}\t'.format(label, ref[1]), end='')
                                if i == 4:
                                   print ('')
                        print ('========== END ==========')
                        self.label_map = []
                elif len(self.label_map) == 0:
                         print ('ERROR! print_label_map() function must be called after generating a batch dataset')

    def shuffle_set(self, set_x, set_y):
            
            # Shuffle
            #It generates a random integer between 0 and 100
            set_seed = random.randint(0, 100)
            #Then, it sets the seed for the random number generator using
            random.seed(set_seed)
            random.shuffle(set_x)
            random.seed(set_seed)
            random.shuffle(set_y)
            return set_x, set_y
        
    def read_images(self, image_file):
            #return np.reshape(cv2.imread(image_file).astype(np.float32)/255,(self.img_size,self.img_size,self.img_channel))
            img = cv2.imread(image_file)
            
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize the image to the desired dimensions (84x84)
            img = cv2.resize(img, (self.img_size, self.img_size))
            # Normalize the pixel values to be between 0 and 1
            img = img.astype(np.float32) / 255.0
            #print("Image shape:", img.shape)
            return img
    def convert_to_tensor(self, np_objects):
            return [tf.convert_to_tensor(obj) for obj in np_objects]

    def generate(self,folder_list,shuffle=False):
            #k images used for meta-train
            k_shot = self.spt_num 
            #k images used for meta-test
            k_query = self.qry_num
            """
            x: The first argument x represents the array or sequence from which to sample.
            k_shot + k_query: The second argument specifies the number of items to sample. Here, k_shot + k_query determines the total number of items to be sampled.
            False: The third argument (False) represents the parameter replace, which signifies whether sampling should be done with replacement (True) or without replacement (False). In this case (False), it means sampling is done without replacement, ensuring each item is sampled only once.
            """
            set_sampler = lambda x: np.random.choice(x, k_shot+k_query, False)
            label_map = []
            images_with_labels = []
            for i, elem in enumerate(folder_list):
                folder = elem[0]
                label = elem[1]
                label_map.append((folder, label))
                #creating image path with label , looping through each folder
                #this creates list of file path for each image from each directory.
                image_with_label = [(os.path.join(folder, image), label) \
                                for image in set_sampler(os.listdir(folder))\
                                    if image.lower().endswith(('.png', '.jpg', '.jpeg'))] #it only does have to read image file.
                #appending each folder file paths
                images_with_labels.append(image_with_label)
            self.label_map.append(label_map)
            if shuffle == True:
                 for i, elm in enumerate(images_with_labels):
                    random.shuffle(elem)

            #slicing the dataset
            def _slice_set(ds):
                spt_x = list()
                spt_y = list()
                qry_x = list()
                qry_y = list()
                
                for i, class_elem in enumerate(ds):
                    #generating random sample for each class image label
                    spt_elem = random.sample(class_elem,self.spt_num)
                    #qry_elem = [(elem[0], elem[1]) for elem in class_elem if elem not in spt_elem]
                    qry_elem = [elem for elem in class_elem if elem not in spt_elem]
                    spt_elem = list(zip(*spt_elem))
                    qry_elem = list(zip(*qry_elem))
                   
                    #This is used to  create x and y image and label matix.label matrix is created by one hot encoding that availabel upto the class
                    spt_x.extend([self.read_images(img) for img in spt_elem[0]])
                    spt_y.extend([tf.one_hot(label, self.n_way) for label in spt_elem[1]])
                    qry_x.extend([self.read_images(img) for img in qry_elem[0]])
                    qry_y.extend([tf.one_hot(label, self.n_way) for label in qry_elem[1]])

                # Shuffle datasets
                spt_x, spt_y = self.shuffle_set(spt_x, spt_y)
                qry_x, qry_y = self.shuffle_set(qry_x, qry_y)
                # convert to tensor
                spt_x, spt_y = self.convert_to_tensor((np.array(spt_x), np.array(spt_y)))
                qry_x, qry_y = self.convert_to_tensor((np.array(qry_x), np.array(qry_y)))
                return spt_x, spt_y, qry_x, qry_y
            return _slice_set(images_with_labels)
        
    def train_batch(self):
             
            # this returns a batch of support set tensor and Quer set tensor
            
            folders = self.metatrain_folders
            # Shuffle root folder in order to prevent repeat
            batch_set = []
            self.label_map = []
            #loop until nuber of task in a batch
            #for each task it creates set of support_x, support_y, query_x, query_y
            for i in range(self.meta_batchsz):
                # function to generate a random array of indices.
                #len(folders): Specifies the range from which indices are chosen
                #self.n_way: Determines the number of indices to be sampled
                sampled_folders_idx = np.array(np.random.choice(len(folders), self.n_way, False))
                np.random.shuffle(sampled_folders_idx)
                #Uses the shuffled indices (sampled_folders_idx) to index the original list of folders and creates a new array 
                sampled_folders = np.array(folders)[sampled_folders_idx].tolist()
                labels = np.arange(self.n_way)
                np.random.shuffle(labels)
                labels = labels.tolist()
                folder_with_label = list(zip(sampled_folders, labels))
                support_x, support_y, query_x, query_y = self.generate(folder_with_label)
                batch_set.append((support_x, support_y, query_x, query_y))
            return batch_set

    def test_batch(self):

            folders = self.metaval_folders
            print ('Sample test batch from {} classes'.format(len(folders))) 

            batch_set = []
            self.label_map = []
                                            
            for i in range(self.meta_batchsz):
                sampled_folders_idx = np.array(np.random.choice(len(folders), self.n_way, False))
                np.random.shuffle(sampled_folders_idx)
                sampled_folders = np.array(folders)[sampled_folders_idx].tolist()
                folder_with_label = []
                labels = np.arange(self.n_way)
                np.random.shuffle(labels)
                labels = labels.tolist()
                folder_with_label = list(zip(sampled_folders, labels))
                support_x, support_y, query_x, query_y = self.generate(folder_with_label)
                batch_set.append((support_x, support_y, query_x, query_y))
            return batch_set
        

if __name__ == '__main__':
    tasks = DataGenerator()
    tasks.mode = 'train'
    for i in range(20):
        batch_set = tasks.train_batch()
        tasks.print_label_map()
        print (len(batch_set))
        time.sleep(5)
