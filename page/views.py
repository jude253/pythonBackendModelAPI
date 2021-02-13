from flask import Blueprint, Flask, render_template, request, send_from_directory
page = Blueprint('templates', __name__, template_folder='templates')


@page.route('/')
def home():
    return render_template('/home.html')

@page.route('/nameDemo/')
def nameDemo():
    return render_template('/nameDemo.html')

@page.route('/textAnalysis/')
def textAnalysis():
    return render_template('/textAnalysis.html')

@page.route('/base/')
def base():
    return render_template('/base.html')

@page.route('/index/')
def index():
    return render_template('/index.html')