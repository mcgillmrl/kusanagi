#!/usr/bin/env python
import os
import sys
import socket
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
#from kusanagi.ghost.algorithms import mc_pilco

UPLOAD_FOLDER = '/home/automation/nikhil/workspace/roughwork_ws/robot_learning_server/uploads' #TODO: Check whether the folder exists
ALLOWED_EXTENSIONS = set(['txt'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/optimize/<task_id>", methods=['GET','POST'])
def optimize(task_id):

    if request.method == "GET":
        # return status of current task_id
        ret = "GET"+str(task_id)

    if request.method == "POST":
        ret = "POST"+str(task_id)

        # check if the post request has the file part
        sys.stderr.write(str(request.files))

        print(request.files)
        if 'file1' not in request.files:
            print('file1 missing')
            return redirect(request.url) 
        if 'file2' not in request.files:
            print('file2 missing')
            return redirect(request.url) 

        f1 = request.files['file1']
        f2 = request.files['file2']

        if f1.filename == '':
            print('file1 not selected')
            return redirect(request.url)
        if f2.filename == '':
            print('file2 not selected')
            return redirect(request.url)

        if f1 and f2 and allowed_file(f1.filename) and allowed_file(f2.filename):
            filename1 = secure_filename(f1.filename)
            filename2 = secure_filename(f2.filename)
            print(filename1, filename2)
            f1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
            f2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))
            return redirect(url_for('optimize', task_id=task_id))

    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__=="__main__":
    app.run(host="0.0.0.0", port=8008)
