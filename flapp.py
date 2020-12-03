from flask import Flask , render_template , request

app = Flask(__name__)

import tensorflow as tf
from PIL import Image
import numpy as np

@app.route('/')
def index():
    return render_template('up.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       

        classes = ['Cardiomegaly',
                    'Emphysema',
                    'Effusion',
                    'Hernia',
                    'Infiltration',
                    'Mass',
                    'Nodule',
                    'Atelectasis',
                    'Pneumothorax',
                    'Pleural_Thickening',
                    'Pneumonia',
                    'Fibrosis',
                    'Edema',
                    'Consolidation']

        im_size = 512
        file_names = []

        f = request.files['file']
        uploaded_files = request.files.getlist("file")

        infer = []
        infer1 = []

        for f in uploaded_files:

            file_names.append(f.filename)

            f = Image.open(f)
            f = f.resize((im_size,im_size))
            f = f.convert("RGB")
            a = np.asarray(f)/255.0
            a = a.reshape(1,im_size,im_size,3)
            infer.append(a)
            infer1.append(1)
        
        infer = np.array(infer).reshape(-1 , im_size,im_size,3)
        infer1 = np.array(infer1).reshape(-1,1)

        inp1 = tf.keras.layers.Input(shape = (im_size,im_size,3))

        o = tf.keras.layers.Conv2D(64,(3,3) , padding="same")(inp1)
        o = tf.keras.layers.Activation('relu')(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.layers.MaxPooling2D((2,2),padding='same')(o)

        o = tf.keras.layers.Conv2D(64,(3,3) , padding="same")(o)
        o = tf.keras.layers.Activation('relu')(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.layers.MaxPooling2D((2,2),padding='same')(o)

        o = tf.keras.layers.Conv2D(64,(3,3) , padding="same")(o)
        o = tf.keras.layers.Activation('relu')(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.layers.MaxPooling2D((2,2),padding='same')(o)

        o = tf.keras.layers.Conv2D(64,(3,3) , padding="same")(o)
        o = tf.keras.layers.Activation('relu')(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.layers.MaxPooling2D((2,2),padding='same')(o)

        o = tf.keras.layers.Conv2D(64,(3,3) , padding="same")(o)
        o = tf.keras.layers.Activation('relu')(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.layers.MaxPooling2D((2,2),padding='same')(o)

        o = tf.keras.layers.Conv2D(64,(3,3) , padding="same")(o)
        o = tf.keras.layers.Activation('relu')(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.layers.MaxPooling2D((2,2),padding='same')(o)

        o = tf.keras.layers.Conv2D(64,(3,3) , padding="same")(o)
        o = tf.keras.layers.Activation('relu')(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.layers.MaxPooling2D((2,2),padding='same')(o)

        o = tf.keras.layers.Flatten()(o)
        o = tf.keras.layers.Dense(128 , activation = 'relu')(o)
        o = tf.keras.layers.BatchNormalization()(o)
        o = tf.keras.layers.Dropout(0.4)(o)
        o = tf.keras.models.Model(inputs=inp1, outputs=o)

        inp2 = tf.keras.layers.Input(shape = (1,))
        t = tf.keras.layers.Dense(128 , activation = 'relu')(inp2)
        t = tf.keras.layers.BatchNormalization()(t)
        t = tf.keras.layers.Dense(128 , activation = 'relu')(t)
        t = tf.keras.layers.BatchNormalization()(t)
        t = tf.keras.layers.Dropout(0.4)(t)
        t = tf.keras.models.Model(inputs=inp2, outputs=t)

        combined = tf.keras.layers.concatenate([o.output, t.output])

        z = tf.keras.layers.Dense(128 , activation = 'relu')(combined)
        z = tf.keras.layers.BatchNormalization()(z)
        z = tf.keras.layers.Dropout(0.4)(z)
        z = tf.keras.layers.Dense(len(classes) , activation = 'sigmoid')(z)

        model = tf.keras.models.Model(inputs=[o.input, t.input], outputs=z)

        model.compile(loss = 'binary_crossentropy' , optimizer = tf.keras.optimizers.SGD() )

        model.load_weights("chest.h5")

        p = (model.predict([infer,infer1])).tolist()[0]

        print(p)
        
        x = []
    
        for i in range(len(p)):
            text = classes[i] + " : " + str(round((p[i]*100),2)) + "%"
            x.append(text)


        tf.keras.backend.clear_session()


        return render_template('out.html' , i = x)

@app.route('/help')
def hel():
    return render_template('help.html')    

if __name__ == "__main__":
    app.run()
