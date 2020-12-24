from flask import Flask, render_template, request
import json
# app reference
from Gender_age_classifier.gender_text_classifier import gender_text_classifier

app = Flask(__name__)

test_entry = '''
Hello I am a human.  I am not fake; I am real.  I realize that my sentences usually start with I am, at least at 
the beginning of a paragraph.  I think that this online course has fried my brain.  I want to stop doing it before
I end up working at a company like the one I worked at before again.  Maybe that is not necessarily going to happen, but I
I am losing faith that this course will help me created my webpage in a way that is fast for me.'''
test_entry2 = '''As of now, after college, I plan on getting my PhD. in either applied mathematics or whichever field is makes most sense to me 3 years down the road. I am not sure that I want to teach later in life as much as I really want to be at the leading edge of research in my field.'''

@app.route('/<string:page_name>/')
def render_static(page_name):
    return render_template('%s.html' % page_name)
# This method executes before any API request

@app.before_request
def before_request():
    print('before API request')
# This method returns students
# list and by default method will be GET

@app.route('/api/textGender', methods=['POST'])
def get_text_gender():
    textInput = request.json['textInput']
    response = gender_text_classifier(textInput)
    return json.dumps(response)

# This is POST method which stores students details.
@app.route('/api/storestudents', methods=['POST'])
def store_student_data():
    return "Student list[POST]"

@app.route('/', methods=['GET', 'POST'])
def form():
    return render_template('index.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():
    print(request.form['say'],request.form['to'])
    return render_template('greeting.html', say=request.form['say'], to=request.form['to'])


# This method executes after every API request.
@app.after_request
def after_request(response):
    return response