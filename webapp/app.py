# -*- coding: utf-8 -*-

from flask import Flask, redirect, url_for, render_template, request, session
from scripts import tabledef
from scripts import forms
from scripts import helpers
from dotenv import load_dotenv
import json
import os
import stripe

import keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
#from mlxtend.preprocessing import minmax_scaling
import h5py
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalAveragePooling2D, Input, BatchNormalization, Multiply, Activation
from keras.optimizers import RMSprop, SGD
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf
import os
from keras.applications.inception_v3 import InceptionV3
import os

import requests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app = Flask(__name__)
app.secret_key = os.urandom(12)  # Generic key for dev purposes only

BASEDIR = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(BASEDIR, './production.env'))

# Get the pub and secret key from the env file and set the stripe api key
pub_key = os.getenv('pub_key')
secret_key = os.getenv('secret_key')
stripe.api_key = secret_key


# Heroku
#from flask_heroku import Heroku
#heroku = Heroku(app)


target_size = (299, 299) #fixed size for InceptionV3 architecture

# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():    
    if not session.get('logged_in'):
        # user not logged in
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    # user logged in    
    user = helpers.get_user()
    if user.payed == 0: 
    # user must pay    
        return render_template('payment.html', pub_key=pub_key)
    else:
    # user paid already    
        #return render_template('product.html', user=user)
        return render_template('home.html', user=user)


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))

# -------- Payments ---------------------------------------------------------- #
@app.route('/purchase', methods=['POST'])
def purchase():
    if session.get('logged_in'):
        customer = stripe.Customer.create(email=request.form['stripeEmail'], source=request.form['stripeToken'])
        charge = stripe.Charge.create(
            customer=customer.id,
            amount=150,
            currency='usd',
            description='The Product')  
        #gets username from session username
        helpers.change_user(payed=1) 
        return redirect(url_for('product')) 
    # user needs to login    
    return redirect(url_for('login'))    


# -------- Product ---------------------------------------------------------- #
@app.route('/product', methods=['GET'])
def product():
    if session.get('logged_in'):
        # user logged in
        user = helpers.get_user()
        if user.payed == 0:
            # user must pay    
            return render_template('payment.html', pub_key=pub_key)
        else:
        # user payed already goto product page
            return render_template('product.html', user=user)
    # user needs to login    
    return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        print("f.filename", f.filename)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        img_path = "./uploads/" + f.filename

        # Make prediction
        print("b4 preds")
        preds = model_predict(img_path, model,  f.filename)
        print("after preds")

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        preds = str(preds).strip('()').split(",")
        result = str(preds[0])
        print("results", result)
        #return result
      
        return result
    return None
# ======== functions ==========================================================#
def load_trained_model():
    model = model_definition()
    #model = load_model('modelFood_6.h5')
    return model

# helper function to load image and return it and input vector
def get_image(path):
    img = image.load_img(path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def model_predict(img_path, model, filename):
    string_class = "apple_pie\ncarrot_cake\ncheesecake\ncup_cakes\ndonuts\ndumplings"
    categories = string_class.splitlines()
    print(categories, len(categories))
   
    print(img_path)
    img, x = get_image(img_path)
    with graph.as_default():	
        probabilities = model.predict(x)[0]
    
    print("PROBS" , probabilities)
    result = [( categories[x], (-np.sort(-probabilities)[i]*100)) for i, x in enumerate(np.argsort(-probabilities)[:5])];
    return result[0]
def model_definition():
    global model
    inc = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
    x = inc.output
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Dropout(.2)(x) # Dropout slows training down
    x = Flatten()(x)
    predictions = Dense(6, kernel_initializer='glorot_uniform', kernel_regularizer=l2(.0005), activation='softmax')(x)
    model = Model(inputs=inc.input, outputs=predictions)
    
    
    model_exists = os.path.isfile('.inceptionv3_3_1.hdf5')
    if not model_exists:
         file_id = '1-4cD6g8ZgOjP_9YbGRFrJ5hU_cZgOTCU'
         
         gdd.download_file_from_google_drive(file_id='1-4cD6g8ZgOjP_9YbGRFrJ5hU_cZgOTCU',
                                        dest_path='./inceptionv3_3_1.hdf5',
                                        unzip=False)
        
    model = load_model('inceptionv3_3_1.hdf5')
    
    #opt = SGD(lr=0.01, momentum=.9)
    model.compile(optimizer="rmsprop", loss='categorical_crossentropy', metrics=['accuracy'])
    global graph
    graph = tf.get_default_graph()    
    return model

from google_drive_downloader import GoogleDriveDownloader as gdd



model = load_trained_model()

print("model loaded successfully")
# ======== Main ============================================================== #
if __name__ == "__main__":
    # thr: address and port changed
    app.run(host='0.0.0.0', port='5000', debug=True, use_reloader=True)
