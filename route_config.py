from flask import Flask, render_template, request, send_from_directory
import json
#import the Deep averaging Neural Network model classifier:

# from Gender_age_classifier.gender_text_classifier import gender_text_classifier
from Name_classifier.nameClassifier import name_classifier
# app reference
app = Flask(__name__)

app.upload_folder = "static/"
app.config['IMAGE_FOLDER'] = "static/images/"

#these are test entries of text input to help me debug the functionality
test_entry = '''
Hello I am a human.  I am not fake; I am real.  I realize that my sentences usually start with I am, at least at 
the beginning of a paragraph.  I think that this online course has fried my brain.  I want to stop doing it before
I end up working at a company like the one I worked at before again.  Maybe that is not necessarily going to happen, but I
I am losing faith that this course will help me created my webpage in a way that is fast for me.'''
test_entry2 = '''As of now, after college, I plan on getting my PhD. in either applied mathematics or whichever field is makes most sense to me 3 years down the road. I am not sure that I want to teach later in life as much as I really want to be at the leading edge of research in my field.'''



#this is for page navigation and image loading:
@app.route('/images/<path:filename>')
def send_image(filename):
    print("here!!!!")
    print(app.config['IMAGE_FOLDER'], filename)
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)

@app.route('/favicon/<path:filename>')
def send_favicon(filename):
    return send_from_directory(app.upload_folder + "favicon/", filename)

@app.route('/<string:page_name>/')
def render_static(page_name):
    print(page_name)
    return render_template('%s.html' % page_name)


@app.route('/', methods=['GET', 'POST'])
def indexHTML():
    return render_template('index.html')

# This method executes before any API request
# @app.before_request
# def before_request():
#     print('before API request')

#this is the api endpoint for classifying the gender of the writer of text, it calls a deep averaging neural network model
# @app.route('/api/textGender', methods=['POST'])
# def get_text_gender():
#     textInput = request.json['textInput']
#     response = gender_text_classifier(textInput)
#     return json.dumps(response)

@app.route('/api/nameGender', methods=['POST'])
def get_name_gender():
    textInput = request.json['nameInput']
    response = name_classifier(textInput)
    return json.dumps(response)

# This method executes after every API request.
# @app.after_request
# def after_request(response):
#     print('after request')
#     return response
