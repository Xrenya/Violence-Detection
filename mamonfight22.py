import numpy as np
from skimage.transform import resize
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Dropout, Flatten, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import Sequential

def video_mamonreader(cv2,filename):
    frames = np.zeros((30, 160, 160, 3), dtype=np.float)
    i=0
    print(frames.shape)
    vc = cv2.VideoCapture(filename)
    if cv2.VideoCapture(filename):
        rval , frame = cv2.imread()
    else:
        rval = False
    frm = resize(frame,(160,160,3))
    frm = np.expand_dims(frm,axis=0)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    print("reading video")
    while i < 30:
        rval, frame = cv2.imread()
        frm = resize(frame,(160,160,3))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frames[i][:] = frm
        i +=1
    return frames

def mamon_videoFightModel(tf,wight='mamonbest947oscombo.hdfs'):
    num_classes = 2
    input_shapes = (160,160,3)
    vg19 = tf.keras.applications.VGG19
    base_model = vg19(include_top=False,weights="imagenet",input_shape=(100,100,3))
    for layer in base_model.layers:
        layer.trainable = False
    model = Sequential()
    num_classes = 2
    cnn = Sequential()
    cnn.add(base_model)
    cnn.add(Flatten())
    model = Sequential()
    model.add(TimeDistributed(cnn,  input_shape=(40, 100, 100, 3)))
    model.add(LSTM(40))
    model.add(Dense(13, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation="sigmoid"))
    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wight)
    model.compile(loss='binary_crossentropy', optimizer= adam, metrics=["accuracy"])
    return model




def mamon_videoFightModel2(tf,wight='mamonbest947oscombo.hdfs'):
    num_classes = 2
    cnn = Sequential()
    #cnn.add(base_model)

    input_shapes=(160,160,3)
    np.random.seed(1234)
    vg19 = tf.keras.applications.VGG19
    base_model = vg19(include_top=False,weights='imagenet',input_shape=(160, 160,3))
    # Freeze the layers except the last 4 layers
    #for layer in base_model.layers:
    #    layer.trainable = False

    cnn = Sequential()
    cnn.add(base_model)
    cnn.add(Flatten())
    model = Sequential()

    model.add(TimeDistributed(cnn,  input_shape=(30, 160, 160, 3)))
    model.add(LSTM(30 , return_sequences= True))

    model.add(TimeDistributed(Dense(90)))
    model.add(Dropout(0.1))

    model.add(GlobalAveragePooling1D())

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(num_classes, activation="sigmoid"))

    adam = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.load_weights(wight)
    rms = RMSprop()

    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

    return model

def pred_fight(model,video,acuracy=0.9):
    pred_test = model.predict(video)
    if pred_test[0][1] >=acuracy:
        return True , pred_test[0][1]
    else:
        return False , pred_test[0][1]
