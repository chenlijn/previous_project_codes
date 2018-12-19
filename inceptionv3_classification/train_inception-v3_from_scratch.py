import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks


class CustomInceptionV3(object):

    def __init__(self, input_shape=(961, 961, 3), class_num = 5):
        self._class_num = class_num
        self._input_shape = input_shape
        return

    def create_custom_inceptionV3(self, pooling_method = 'avg'):
        """create an custom InceptionV3 network"""
    
        input_tensor = Input(shape=self._input_shape)
        self.base_model = InceptionV3(weights=None, include_top = False, input_tensor = input_tensor, pooling = pooling_method)
        #pretrained_model='pretrained_model/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        #self.base_model.load_weights(pretrained_model)
    
        # add a fully-connected and the logistic layers

        dense_input = Dropout(rate=0.3)(self.base_model.output)
        dense_feature = Dense(1024, activation='relu')(dense_input)
        dense_feature2 = Dropout(rate=0.3)(dense_feature)
        predictions = Dense(self._class_num, activation='softmax')(dense_feature2)
    
        return Model(inputs=self.base_model.input, outputs=predictions)
    
    
    def create_adam_optimizer(self, lr=0.001):
        return keras.optimizers.Adam(lr=lr, beta_1 = 0.9, beta_2=0.999, epsilon=None)

    def preprocess_funct(self, image):
        image -= 127.5
        image /= 127.5
        return image    
    
    def train_network(self, model, train_data_directory, val_data_directory, train_num, val_num, class_names=['0', '1', '2', '3', '4'], batch_size=32, epoch_num=200, lr=0.001):

        ## load pretrained weights
        #model.load_weights('saved_models/model_010-0.24.h5')
    
        # generate training data
        #train_datagen = ImageDataGenerator(rescale=1/255.0) # no augmentation, only generating batches of images
        train_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_funct) # no augmentation, only generating batches of images
        train_generator = train_datagen.flow_from_directory(train_data_directory, 
                                                            target_size=(self._input_shape[0], self._input_shape[1]),
                                                            batch_size=batch_size, 
                                                            classes=class_names,
                                                            class_mode='categorical')

        ##val_datagen = ImageDataGenerator(rescale=1/255.0) # no augmentation, only generating batches of images
        val_datagen = ImageDataGenerator(preprocessing_function=self.preprocess_funct) # no augmentation, only generating batches of images
        val_generator = val_datagen.flow_from_directory(val_data_directory, 
                                                            target_size=(self._input_shape[0], self._input_shape[1]),
                                                            batch_size=batch_size, 
                                                            classes=class_names,
                                                            class_mode='categorical')

        for i in range(10):
            images, labels = train_generator.next()
            print("labels: ")
            print labels

        ### define the callback functions
        #file_path = 'training/model_{epoch:03d}-{val_loss:.2f}.h5'
        #checkpointer = callbacks.ModelCheckpoint(file_path, monitor='val_loss', verbose=0, save_best_only=True)
        #early_stopper = callbacks.EarlyStopping(monitor='loss', patience=10)
        #lr_reducer = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3)

        #### first: freeze all convolutional InceptionV3 layers, train only the top layers
        ##for layer in self.base_model.layers:
        ##    print layer, '    ', layer.trainable
        ##    layer.trainable = True


        #model.compile(optimizer=self.create_adam_optimizer(lr), loss='categorical_crossentropy', metrics=['accuracy'])
        ##model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

        ##test
        #print("model evaluate..")
        ##print(model.evaluate_generator(val_generator, steps=val_num/batch_size))
        #print "fiiting ... "

        ## train the top layers for a few epochs
        #model.fit_generator(train_generator, 
        #                    steps_per_epoch=train_num/batch_size,
        #                    epochs=epoch_num, # for initializing
        #                    callbacks=[checkpointer, early_stopper, lr_reducer],
        #                    validation_data=val_generator, 
        #                    validation_steps=val_num/batch_size) 
        #model.save('saved_models/incept-v3_whole_after_top_layers_trained.h5')

        ### second, freeze the bottom 249 layers and train the remaining top layers
        ##first_n =165 
        ##for i, layer in enumerate(self.base_model.layers):
        ##    #print i, layer.name
        ##    #layer.trainable = True
        ##    if i <= first_n:
        ##        layer.trainable = False
        ##    else:
        ##        layer.trainable = True

        ##model.compile(optimizer=self.create_adam_optimizer(0.001), loss='categorical_crossentropy')
        ###model.compile(optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy')
        ##model.fit_generator(train_generator, 
        ##                    steps_per_epoch=train_num/batch_size,
        ##                    epochs=epoch_num, 
        ##                    callbacks=[checkpointer, early_stopper, lr_reducer],
        ##                    validation_data=val_generator, 
        ##                    validation_steps=val_num/batch_size) 

        ### save the whole model
        ##model.save('saved_models/incept-v3_whole_model_final.h5')


def main():

    train_data_dir = '/root/Shared/ML_Projects_Data/Apollo/chenlijian/apollo_disc_train_test/train_apollo_disc_v2/disc'
    val_data_dir = '/root/Shared/ML_Projects_Data/Apollo/chenlijian/apollo_disc_train_test/test_apollo_disc_v2/disc'
    train_num = 27266 
    val_num = 577

    my_inceptionV3 = CustomInceptionV3(input_shape=(961,961,3), class_num=5)
    model = my_inceptionV3.create_custom_inceptionV3()
    #model = load_model('saved_models/model_011-0.30.h5')
    my_inceptionV3.train_network(model, train_data_dir, val_data_dir, train_num, val_num, batch_size=4, epoch_num = 200, lr=0.0001)

if __name__=='__main__':
    main()











