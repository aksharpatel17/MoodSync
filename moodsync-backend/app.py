from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
import json
import boto3
from flask_cors import CORS 
from dotenv import load_dotenv
from analyze_mood import predict_emotion


app = Flask(__name__)
load_dotenv()

CORS(app) 


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")

aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

region_name = os.getenv("REGION_NAME")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def invoke_lambda_function(filename, data):
    lambda_client = boto3.client('lambda', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=region_name)


    payload = {
        'body': json.dumps({'data': data}) 
    }

    response = lambda_client.invoke(
        FunctionName='get_playlist',
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )

    result = json.loads(response['Payload'].read().decode('utf-8'))
    return result

@app.route('/upload', methods=['POST'])
def upload_file():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        data = predict_emotion(filepath)
        lambda_result = invoke_lambda_function(filename, data)
        print(lambda_result)

        return jsonify({'filename': filename, 'lambda_result': lambda_result})

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
