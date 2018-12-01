#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 18:15:13 2018

@author: fadlimuharram
"""

# import the necessary packages
from time import time
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2D

from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, Activation
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
import flask
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
from keras.models import load_model



# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
klasifikasi = None
train_set = None
test_set = None
datanya = None
jumlahKelas = None

'''development atau production atau initial'''

MODENYA = None 
productionEpochnya = None

print('Input Dengan Menggunakan Model Yang Telah Tersedia')
print('[0] Tidak')
print('[1] Ya')
isLoadedDariModel = int(input('pilihan: '))
print(isLoadedDariModel, type(isLoadedDariModel))
print(isLoadedDariModel != 0)
if isLoadedDariModel != 1 and isLoadedDariModel != 0 :
    raise ValueError('Error: Mohon Pilih 0 atau 1')

if isLoadedDariModel == 1:
    isLoadedDariModel = True
    print('Masukan Jumlah Epoch Sebelumnya')
    productionEpochnya = int(input('jumlah: '))
elif isLoadedDariModel == 0:
    isLoadedDariModel = False
    print('Pilih Mode Training')
    print('[0] initial')
    print('[1] development')
    print('[2] Production')
    MODENYA = int(input('pilihan: '))
    if MODENYA == 0:
        MODENYA = 'initial'
    elif MODENYA == 1:
        MODENYA = 'development'
    elif MODENYA == 2:
        MODENYA = 'production'
    else:
        raise ValueError('Error: pilih mode 0 - 2')
    
    if MODENYA == 'development' or MODENYA == 'production':
        print('Pilih Jumlah Epoch Yang Di Inginkan')
        productionEpochnya = int(input('jumlah: '))
        
else:
    raise ValueError('Error: pilih 0 atau 1 saja')


#isLoadedDariModel = True
#productionEpochnya = 5

IPNYA = '192.168.43.70'
PORTNYA = 5050


LOKASI_TRAINING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/training_set'
LOKASI_TESTING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/testing_set'

#LOKASI_TRAINING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/training_set'
#LOKASI_TESTING = '/Users/fadlimuharram/Documents/cnn/tomat/segmentTambahan/testing_set'

LOKASI_UPLOAD = 'upload'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

PENYAKIT_TANAMAN = {
        0: {
                "nama":"Bacterial Spot",
                "gejala":'''
                <ul>
                    <li style="text-align: justify;">Foliar symptoms appear as dark, water-soaked, circular spots less than 3 millimeters in diameter.</li>
                    <li style="text-align: justify;">These spots become angular and the surfaces may appear greasy, with translucent centers and black margins.</li>
                    <li style="text-align: justify;">The centers of these lesions soon dry and crack, and yellow halos may surround the lesions. During periods of high moisture (heavy rain, fog or dew) leaves will turn chlorotic and may eventually become blighted.</li>
                    <li style="text-align: justify;">Disease can develop on all above-ground parts of the plant, with lesions tending to be more numerous on young foliage.</li>
                    <li style="text-align: justify;">Fruit infection begins as small, black, raised spots which may be surrounded by a white, greasy-appearing halo.</li>
                    <li style="text-align: justify;">Fruit lesions typically enlarge to four to five millimeters in diameter and can be dark-brown with raised margins and sunken, tan centers or tan with raised margins and sunken, dark-brown centers. Fruit lesions often appear scab-like or corky.</li>
                </ul>
                ''',
                "penangan":'''
                <p style="text-align: justify;">Sow only seed that has been tested and certified free of these bacteria and ensure that transplants are disease-free. Copper sprays can provide moderate levels of protection, although copper-resistant strains have become more common. Avoid overhead irrigation. Rotate to non-host crops and control weeds and volunteer plants. Good sanitation practices, including cleaning of equipment and plowing under all plant debris immediately after harvest, can reduce losses from this disease. It is valuable to know which race of bacterial spot predominates in an area, as resistant tomato varieties may be available.</p>
                '''
            },
        1: {
                "nama":"Late Blight",
                "gejala":'''
                <ul>
                    <li style="text-align: justify;">The first symptom of late blight is a bending down of petioles of infected leaves.</li>
                    <li style="text-align: justify;">Leaf and stem lesions manifest as large, irregular, greenish, water-soaked patches. These patches enlarge and turn brown and paper-like.</li>
                    <li style="text-align: justify;">During wet weather, Phytophthora infestans will grow and sporulate from lesions on abaxial leaf surfaces.</li>
                    <li style="text-align: justify;">Rapid blighting of foliage may occur during moist, warm periods.</li>
                    <li style="text-align: justify;">Entire fields can develop extensive foliar and fruit damage. Fruit lesions manifest as large, firm, irregular, brownish-green blotches.</li>
                    <li style="text-align: justify;">Surfaces of fruit lesions are rough and greasy in appearance.</li>
                </ul>
                ''',
                "penangan":'''
                <p style="text-align: justify;">Implement a late blight forecasting system in conjunction with an effective spray program to control late blight. Avoid planting on land previously cropped to potatoes or near a potato field because P. infestans is also a pathogen of potato. In protected culture, maintaining lower humidity will discourage infection and disease development.</p>
                '''
            },
        2: {
                "nama":"Septoria Leaf Spot",
                "gejala":'''
                <ul>
                    <li style="text-align: justify;">This fungus can cause damping-off, crown and root rot, and fruit rot.</li>
                    <li style="text-align: justify;">The first symptom on seedlings is a dark-brown lesion at or below the soil line.</li>
                    <li style="text-align: justify;">Stem tissue is invaded completely, causing seedlings to quickly damp-off and die. On older plants, developing lesions girdle stems, causing plants to wilt without a change in foliage color.</li>
                    <li style="text-align: justify;">These lesions may continue expanding to cause root rot below the soil line and may also extend several centimeters above it.</li>
                    <li style="text-align: justify;">If moisture is adequate, white mycelium grows over lesion surfaces and tan sclerotia (one to two millimeters in diameter) are readily formed. Severely infected plants may eventually die. Fruit that contact Sclerotium rolfsii are invaded quickly, resulting in sunken, yellowish lesions with ruptured epidermises. White mycelium grows from fruit lesions and sclerotia form on lesion surfaces.</li>
                </ul>
                ''',
                "penangan":'''
                <p style="text-align: justify;">Regulate soil moisture levels and deep-plow plant residues to reduce losses from southern blight. Implement a sanitation program that includes removal or burning of all infected plants. Apply fungicides, fumigate soil and rotate from tomato to non-host crops like corn, grain sorghum and wheat for three years to reduce losses from White mycelium and sclerotia on lower stem. southern blight.</p>
                '''
            },
        3: {
                "nama":"Spider Mites",
                "gejala":'''
                <p style="text-align: justify;">Spider mites are among the most ubiquitous of pests, attacking a wide variety of field, garden, greenhouse, nursery, and ornamental plants, as well as several weed species. Infestations of two-spotted spider mites result in the bleaching and stippling of leaves. Severe infestations may cause entire leaves to become bronzed, curled, and completely enveloped in sheets of webbing. Damage to the foliage may result in leaf drop and reduction in the overall vitality of the plant. When a leaf or branch is tapped over a white sheet of paper, the mites appear as small specks that resemble dust or pepper and may be seen to move.</p>
                ''',
                "penangan":'''
                <p style="text-align: justify;"><span style="color: #000000;">1.&nbsp;<strong>Knock mites off plants with water.</strong>&nbsp;Spraying with a strong stream of water (particularly the undersides of leaves) will provide some control. Spray plants frequently to control future buildups. For severe infestations, affected plants or plant parts can be removed and destroyed. There are several natural predators that feed on spider mites. The use of chemical insecticides to control other garden pests can result in the death of these beneficial insects and a subsequent increase in the population of spider mites.</span></p>
                <p style="text-align: justify;"><span style="color: #000000;">2.&nbsp;<strong>Use insecticidal&nbsp;soap.</strong>&nbsp;Insecticidal&nbsp;soaps specially formulated to kill insects and not damage plants are effective if used frequently until the problem is under control.</span></p>
                <p style="text-align: justify;"><span style="color: #000000;">3.&nbsp;<strong>Use superior horticultural&nbsp;oil&nbsp;sprays.</strong>&nbsp;Highly refined oils sold as superior or horticultural oils are also very effective in controlling spider mites. The oil suffocates the mites. Unlike dormant oils, these oils are highly refined and under proper conditions, can be applied to plant foliage without damage. Follow label directions to avoid damage to some plants that may be sensitive. Superior oils are considered nontoxic and are less likely to kill beneficial insects.</span></p>
                <p style="text-align: justify;"><span style="color: #000000;">4.&nbsp;<strong>Use chemical insecticides or miticides.</strong>&nbsp;A very safe product made from the seeds of a tropical tree is called&nbsp;Neem. It is commercially available under the name Margosan-O. Other chemical controls include&nbsp;malathion,&nbsp;bifenthrin,&nbsp;cyfluthrin, and kelthane. Be sure to follow all label directions when using pesticides. Many pesticides are very harmful to bees and fish when used improperly.</span></p>
                '''
            }
        }

print(PENYAKIT_TANAMAN[0])
def hitungGambar(path):
    count = 0
    for filename in os.listdir(path):
        if filename != '.DS_Store':
            count = count + len(os.listdir(path+'/'+filename))
    return count

def hitunKelas():
    global LOKASI_TRAINING, LOKASI_TESTING, PENYAKIT_TANAMAN
    kelasTraining = 0
    kelasTesting = 0
    
    for filename in os.listdir(LOKASI_TRAINING):
        if filename != '.DS_Store':
            kelasTraining = kelasTraining + 1
            
    for filename in os.listdir(LOKASI_TESTING):
        if filename != '.DS_Store':
            kelasTesting = kelasTesting + 1
            
    if kelasTesting == kelasTraining and kelasTraining == len(PENYAKIT_TANAMAN) and kelasTesting == len(PENYAKIT_TANAMAN):
        return kelasTraining
    else:
        raise ValueError('Error: Kelas Training tidak sama dengan Kelas Testing')
        






app.config['UPLOAD_FOLDER'] = LOKASI_UPLOAD
app.config['STATIC_FOLDER'] = LOKASI_UPLOAD
jumlahKelas = hitunKelas()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
def load_model_klasifikasi():
    global klasifikasi, train_set, test_set, datanya, kelasnya, LOKASI_TRAINING, LOKASI_TESTING
    global MODENYA, productionEpochnya, isLoadedDariModel
    # initialising the cnn
    klasifikasi = Sequential()
    
    
  
    
    
    klasifikasi.add(Convolution2D(12,    # number of filter layers
                        5,    # y dimension of kernel (we're going for a 3x3 kernel)
                        5,    # x dimension of kernel
                        input_shape=(64, 64, 3),
                        init='he_normal'))
    # Lets activate then pool!
    klasifikasi.add(Activation('relu'))
    klasifikasi.add(MaxPooling2D(pool_size=(2,2)))
    klasifikasi.add(Convolution2D(25,    # number of filter layers
                            5,    # y dimension of kernel (we're going for a 3x3 kernel)
                            5,    # x dimension of kernel
                            init='he_normal'))
    # Lets activate then pool!
    klasifikasi.add(Activation('relu'))
    klasifikasi.add(MaxPooling2D(pool_size=(2,2)))
    
    
    
    #Flatten
    klasifikasi.add(Flatten())
    klasifikasi.add(Dense(180, activation = 'relu', init='he_normal'))
    klasifikasi.add(Dropout(0.5))
    klasifikasi.add(Dense(100, activation = 'relu', init='he_normal'))
    klasifikasi.add(Dropout(0.5))
  
    
    
    klasifikasi.add(Dense(jumlahKelas, activation='softmax',init='he_normal'))
    print("Full Connection Between Hidden Layers and Output Layers Completed")

    
    
    
    
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_set = train_datagen.flow_from_directory(
            LOKASI_TRAINING,
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            LOKASI_TESTING,
            target_size=(64, 64),
            batch_size=8,
            class_mode='categorical')
    
    if isLoadedDariModel == True:
        namaFilenya = "modelKlasifikasi" + str(productionEpochnya) +".h5"
        if os.path.exists(namaFilenya) :
            klasifikasi = load_model(namaFilenya)
            datanya = klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
        else:
            raise ValueError('Error: File Tidak Ada Harap Lakukan Training Terlebih Dahulu Sebelum Menggunakan Model')
    else:
        # kompile cnn
        klasifikasi.compile(optimizer='adam', loss='categorical_crossentropy', metrics= ['accuracy'])
        print(klasifikasi.summary())
        print("Compiling Initiated")
        if MODENYA == 'development':
            
            datanya = klasifikasi.fit_generator(
                train_set,
                steps_per_epoch=50,
                epochs=productionEpochnya,
                validation_data=test_set,
                validation_steps=30)
            
            klasifikasi.save("modelKlasifikasi" + str(productionEpochnya) +".h5")
            
        elif MODENYA == 'production' :
            
            tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
            
            datanya = klasifikasi.fit_generator(
                train_set,
                steps_per_epoch=hitungGambar(LOKASI_TRAINING),
                epochs=productionEpochnya,
                validation_data=test_set,
                validation_steps=hitungGambar(LOKASI_TESTING),
                callbacks=[tensorboard]
                )
            klasifikasi.save("modelKlasifikasi" + str(productionEpochnya) +".h5")
            
        elif MODENYA == 'initial' :
            
            datanya = klasifikasi.fit_generator(
                train_set,
                steps_per_epoch=5,
                epochs=1,
                validation_data=test_set,
                validation_steps=2)
        gambarHasilLatih()
        
    klasifikasi._make_predict_function()
    print("Compiling Completed")


def gambarHasilLatih():
    global datanya
    # Plot training & validation accuracy values
    plt.plot(datanya.history['acc'])
    plt.plot(datanya.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    # Plot training & validation loss values
    plt.plot(datanya.history['loss'])
    plt.plot(datanya.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

        
@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    global train_set, klasifikasi, IPNYA, PORTNYA, LOKASI_UPLOAD, PENYAKIT_TANAMAN
    print('-------------')
    print(request.method)
    print(request.files)
    print('-------------')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save('static/' + os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename)
            
            lokasiTest = LOKASI_UPLOAD + '/' + filename
          
            test_image = image.load_img('static/' + lokasiTest, target_size = (64, 64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = klasifikasi.predict(test_image).tolist()
            '''result = pd.Series(result).to_json(orient='values')'''
            print(train_set.class_indices)
            '''return redirect(url_for('uploaded_file',filename=filename))'''
            print(result)
            
            hasil = {}
            dataJSON = {}
            allProba = {}
            loop = 0
            
            for cls, val in train_set.class_indices.items():
                '''hasil[cls] = result[0][train_set.class_indices[cls]]'''
                
                proba = result[0][train_set.class_indices[cls]]
                allProba[cls] = proba
                print(proba)
                if (proba > 0.0) and (proba <= 1.0) :
                    print('valnya : ' + str(val))
                    '''hasil.update({'datanya':{PENYAKIT_TANAMAN[val]},'probability':proba})'''
                    hasil["proba" + str(loop)] = PENYAKIT_TANAMAN[val]
                    hasil["proba" + str(loop)]['probability'] = proba
            
                    loop = loop + 1
            print(hasil)
            dataJSON['Debug'] = allProba
            dataJSON['penyakit'] = hasil
            dataJSON['uploadURI'] = 'http://' + IPNYA + ':' + str(PORTNYA) + url_for('static',filename=lokasiTest)
            
            return flask.jsonify(dataJSON)
        

    else:
        
        return '''
            <!doctype html>
            <title>Upload new File</title>
            <h1>Upload new File</h1>
            <form method=post enctype='multipart/form-data'>
              <p><input type='file' name='file'>
                 <input type='submit' value='Upload'>
            </form>
            '''
  
        
if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model_klasifikasi()
    app.run(host=IPNYA, port=PORTNYA,debug=True)