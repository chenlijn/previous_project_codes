#it is for sorting the data

import os
import shutil
import numpy as np
import cv2
import random


class sort_data:

    def __init__(self):
        self.cwd = os.getcwd()

    def list_data_to_txt(self, data_path, txt_file):
        with open(txt_file, 'w+') as f:
            data_list = os.listdir(data_path)
            for data in data_list:
                if ('.db' in data) or ('.DS_Store' in data):
                    continue
                check_name = data_path + '/' + data 
                if os.path.getsize(check_name) <= 1*1024:
                    #print check_name
                    continue
                #save_name = data_path + '/' + data.replace(' ', '').replace('(','_').replace(')','') + '\n'
                save_name = check_name + '\n'
                f.write(save_name)

    #def list_folder_data(self, data_path):
    def clean_data(self, data_list_file, error_list_file):
        with open(data_list_file,'r') as df:
            data_list = df.readlines()
        with open(error_list_file, 'r') as ef:
            error_list = ef.readlines()

        #remove errors
        #a = '/ill/mydriasis_diffuse_light/100282.jpg\r\n'
        
        for error_data in error_list:
            for data in data_list:
                if error_data.strip('\r\n') in data.strip('\n'):
                    print 'match'
                    data_list.remove(data)

        #store the cleaned data
        save_file = self.cwd + '/cleaned_' + data_list_file.strip('\n').split('/')[-1]
        with open(save_file, 'w+') as sf:
            sf.writelines(data_list)

    
    def append_data_label_in_txtfile(self, data_txt_file, dst_path, name_mapping_txt):
        with open(name_mapping_txt, 'r') as nf:
            name_mappings = nf.readlines()

        filename = data_txt_file.split('/')[-1]
        savefile = dst_path + '/' + filename

        #get label
        lable = ''
        for mapping in name_mappings:
            name = mapping.split(' ')[0]
            label = mapping.strip('\n').split(' ')[-1]
            if name in filename:
                break

        with open(data_txt_file, 'r') as df:
            data_list = df.readlines()
            with open(savefile, 'w+') as sf:
                for line in data_list:
                    sf.write(line.strip('\n') + ' ' + label + '\n')

        return


    #split the data into train, val and test sets
    def split_data_to_train_val_test(self, data_list_file, train_percnt, val_percnt, max_num, dst_path):
        with open(data_list_file, 'r') as df:
            data_list = df.readlines()

        num = len(data_list) 
        filename = data_list_file.split('/')[-1].split('.')[0]
        train_txt = filename + '_train.txt'
        val_txt = filename + '_val.txt'
        test_txt = filename + '_test.txt'

        if num > max_num:
            train_num = max_num * train_percnt 
            val_num = max_num * (train_percnt + val_percnt)
            test_num = max_num
        else:
            train_num = num * train_percnt 
            val_num = num * (train_percnt + val_percnt)
            test_num = num

        with open(dst_path + '/' + train_txt, 'w+') as trf:
            with open(dst_path + '/' + val_txt, 'w+') as vf:
                with open(dst_path + '/' + test_txt, 'w+') as ttf:
                    i = 0
                    for i in range(num):
                        if i < train_num:
                            trf.write(data_list[i])
                        elif (i >= train_num + 10) and (i < val_num): 
                            vf.write(data_list[i]) 
                        elif (i >= val_num + 10) and (i <= test_num):
                            ttf.write(data_list[i])

        return  


                



if __name__ == "__main__":

    sorter = sort_data()

    ##------------------------
    ##sort and clean all data
    ##------------------------
    #data_type = 'healthy'
    #data_type = 'after_surgery'
    #data_type = 'ill'
    ##data_path = '/mnt/lijian/mount_out/zhongshan_ophthalmology_data_20171214/' + data_type
    #data_path = '/mnt/lijian/mount_out/zhongshan_ophthalmology_cataract_images_20170906/' + data_type
    #error_file = sorter.cwd + '/error_files/' + data_type + '.txt'
    #folders = os.listdir(data_path)
    #for folder in folders:
    #    folder_path = data_path + '/' + folder 
    #    #txt_file_name = sorter.cwd + '/new_' + data_type + '_' + folder + '.txt'
    #    txt_file_name = sorter.cwd + '/' + data_type + '_' + folder + '.txt'
    #    sorter.list_data_to_txt(folder_path, txt_file_name)

    #    #clean the data
    #    sorter.clean_data(txt_file_name, error_file)



    ##--------------------------------------
    ##label the data
    ##-------------------------------------- 
    ##illness data
    #src_path = os.getcwd() + '/cleaned_original_data'
    #dst_path = os.getcwd() + '/labeled_data'
    #txt_files = os.listdir(src_path)
    #name_mapping_txt = os.getcwd() + '/cataract_illness_name_mapping.txt'

    ###imaging types data
    ##src_path = os.getcwd() + '/cleaned_original_data'
    ##dst_path = os.getcwd() + '/imaging_types/labeled_data'
    ##txt_files = os.listdir(src_path)
    ##name_mapping_txt = os.getcwd() + '/imaging_types/imaging_types_name_mapping.txt'
    #for txt in txt_files:
    #    sorter.append_data_label_in_txtfile(src_path + '/' + txt, dst_path, name_mapping_txt)

    #-----------------------------------------------------
    #split the data into train, validataion and test sets
    #-----------------------------------------------------
    ##the illness data
    ##src_path = os.getcwd() + '/cleaned_original_data'
    #src_path = os.getcwd() + '/labeled_data_to_split'
    #dst_path = os.getcwd() + '/train_val_split'

    ##data to be split
    #train_percnt = 0.6
    #val_percnt = 0.2
    #test_percnt = 0.2
    #max_num = 2300

    #the illness data
    #src_path = os.getcwd() + '/cleaned_original_data'
    src_path = os.getcwd() + '/labeled_data_to_split'
    dst_path = os.getcwd() + '/train_val_split'

    #data to be split
    train_percnt = 0.6
    val_percnt = 0.2
    test_percnt = 0.2
    max_num = 10000
    txt_files = os.listdir(src_path)
    for txt in txt_files:
        sorter.split_data_to_train_val_test(src_path + '/' + txt, train_percnt, val_percnt, max_num, dst_path)    




    
