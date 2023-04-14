from flask import Flask, jsonify, request, render_template, send_from_directory
from VITON_Infer import InferVITON
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
import argparse
import json
import binascii
import base64
import os
import re

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploaded'
app.config['IMAGE_EXTS'] = [".png", ".jpg", ".jpeg", ".gif", ".tiff"]
app.config['JSON_SORT_KEYS'] = False
ALLOWED_EXTENSIONS = re.compile(r'jpg|png|jpeg', re.IGNORECASE)
counter = 0
processing_buffer = {}


def decode_image(im_b64):
    im_bytes = base64.b64decode(im_b64)   # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file).convert("RGB")   # img is now PIL Image object
    return img

def encode(x):
    return binascii.hexlify(x.encode('utf-8')).decode()

def decode(x):
    return binascii.unhexlify(x.encode('utf-8')).decode()

@app.route("/get-processed-image", methods=["GET"])
def get_processed_image():
    if processing_buffer:
        im_b64 = viton.infer(processing_buffer['img_path'], processing_buffer['cloth_path'])
        processing_buffer.clear()
        response = {"response": 'data:image/jpeg;base64,' + im_b64.decode()}
    else:
        response = jsonify({'response': "No image found on server for processing!"})
        response.status_code = 500
    return response

@app.route("/upload-image", methods=["POST"])
def upload_image():
    """
    requires two images: cloth_img and img
    """

    request_content = json.loads(request.data.decode('utf-8'))
    if 'uploaded_person_img' not in request_content or 'cloth_img' not in request_content:
        response = jsonify({'message': 'img or cloth_img part missing in the request!!'})
        response.status_code = 400
        return response
    if request_content['uploaded_person_img'] == '' or request_content['cloth_img'] == '':
        response = jsonify({'message': 'img or cloth_img not selected for processing!!'})
        response.status_code = 400
        return response
    
    person_image_content = request_content['uploaded_person_img']  # data:image/jpeg;base64
    cloth_image_content = request_content['cloth_img']

    person_extension = person_image_content.split(',')[0].replace('data:','').replace(';base64', '')
    cloth_extension = cloth_image_content.split(',')[0].replace('data:','').replace(';base64', '')

    global counter
    if (ALLOWED_EXTENSIONS.search(person_extension) and 
        ALLOWED_EXTENSIONS.search(cloth_extension)):
        person_image_b64 = person_image_content.split(',')[-1].strip()
        cloth_image_b64 = cloth_image_content.split(',')[-1].strip()
        person_image = decode_image(person_image_b64)
        cloth_image = decode_image(cloth_image_b64)

        person_image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'person_image_' + str(counter) + '.jpg')
        cloth_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cloth_' + str(counter) + '.jpg')
        person_image.save(person_image_filepath)
        cloth_image.save(cloth_filepath)
        
        counter += 1

        processing_buffer['img_path'] = person_image_filepath
        processing_buffer['cloth_path'] = cloth_filepath
        response = jsonify({'message': 'File succesfully uploaded!!'})
        response.status_code = 200
        return response
    else:
        resp = jsonify({'message': 'Allowed file types are [jpg, jpeg, png]'})
        resp.status_code = 415
        return resp

@app.route('/cdn/<path:filepath>')
def download_file(filepath):
    dir,filename = os.path.split(decode(filepath))
    return send_from_directory(dir, filename, as_attachment=False)

@app.route('/')
def home():
    # return "Server Active!"
    root_dir = app.config['ROOT_DIR']
    image_paths = []
    for root,dirs,files in os.walk(root_dir):
        for file in files:
            if any(file.endswith(ext) for ext in app.config['IMAGE_EXTS']):
                image_paths.append(encode(os.path.join(root,file)))
    return render_template('index.html', paths=image_paths)

if __name__ == "__main__":
    # load model and any initial items required
    parser = argparse.ArgumentParser('Usage: %prog [options]')
    parser.add_argument('root_dir', help='Gallery root directory path')
    parser.add_argument('-l', '--listen', dest='host', default='127.0.0.1', \
                                    help='address to listen on [127.0.0.1]')
    parser.add_argument('-p', '--port', metavar='PORT', dest='port', type=int, \
                                default=5000, help='port to listen on [5000]')
    args = parser.parse_args()
    app.config['ROOT_DIR'] = args.root_dir
    viton = InferVITON()
    app.run(host=args.host, port=args.port, debug=True)
    
