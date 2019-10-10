# -*- coding: utf-8 -*-

from scripts import tabledef
from scripts import forms
from scripts import helpers
from flask import Flask, redirect, url_for, render_template, request, session
from dotenv import load_dotenv
import json
import sys
import os
import stripe

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
        return render_template('product.html', user=user)


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


# ======== Main ============================================================== #
if __name__ == "__main__":
    # thr: address and port changed
    app.run(host='0.0.0.0', port='5000', debug=True, use_reloader=True)
