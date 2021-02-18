from flask import Flask, render_template, request, send_from_directory
import json
from flask_cors import cross_origin
#import the Deep averaging Neural Network model classifier:

from Text_classifiers.text_analysis_classifiers import text_analysis_classifier
from Name_classifier.nameClassifier import name_classifier
from page.views import page

# app reference
app = Flask(__name__)
app.register_blueprint(page)
app.upload_folder = "static/"
app.config['IMAGE_FOLDER'] = "static/images/"

#these are test entries of text input to help me debug the functionality

#this is for page navigation and image loading:
@app.route('/images/<path:filename>')
def send_image(filename):
    # print("here!!!!")
    # print(app.config['IMAGE_FOLDER'], filename)
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/favicon/<path:filename>')
def send_favicon(filename):
    return send_from_directory(app.upload_folder + "favicon/", filename)

# @app.route('/<string:page_name>/')
# def render_static(page_name):
#     # print(page_name)
#     return render_template('%s.html' % page_name)


# @app.route('/', methods=['GET', 'POST'])
# def indexHTML():
#     return render_template('index.html')

# this is the api endpoint for classifying the gender of the writer of text, it calls a deep averaging neural network model


@app.route('/api/textAnalysis', methods=['POST'])
@cross_origin()
def get_text_analysis():
    textInput = request.json['textInput']
    response = text_analysis_classifier(textInput)
    return json.dumps(response)

@app.route('/api/nameGender', methods=['POST'])
@cross_origin()
def get_name_gender():
    textInput = request.json['nameInput']
    response = name_classifier(textInput)
    return json.dumps(response)
