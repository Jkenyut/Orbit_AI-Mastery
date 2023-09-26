# web flask library
import os
from flask import Flask, render_template, url_for, request
from werkzeug.utils import secure_filename
# fungsi buatan sendiri -> menyesuaikan
from employee_analysis import *


# file upload
UPLOAD_FOLDER = 'static/video-uploaded/'
ALLOWED_EXTENSIONS = {'mp4'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# validation file upload


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Halaman awal


@app.route('/', methods=['POST', 'GET'])
def index():
    video, data = None, None
    # jika ada request file
    if request.method == 'POST':
        video = request.files['video']
    # jika file yang dikirim berformat mp4
    if video and allowed_file(video.filename):
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        print(str(UPLOAD_FOLDER+video.filename))
        print(str(video.filename))
        # menampilkan hasil dari analisis
        data = analysis(str(UPLOAD_FOLDER+video.filename), str(video.filename))
    return render_template('index.html', video=data)


if __name__ == '__main__':
    app.run(debug=True)
