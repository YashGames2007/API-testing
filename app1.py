from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/api/')
def api_endpoint():
    # Required Parameters
    # image: string format of camera image inputs
    # s1: Time since which Signal 1 is turned Off
    # s2: Time since which Signal 2 is turned Off
    # s3: Time since which Signal 3 is turned Off
    # s4: Time since which Signal 4 is turned Off
    # cam_id: showing camera 1 or camera 2
    image = str(request.args.get('image', '0'))  # Default to '0' if 'image' is not provided
    s1 = int(request.args.get('s1', 0))
    s2 = int(request.args.get('s2', 0))
    s3 = int(request.args.get('s3', 0))
    s4 = int(request.args.get('s4', 0))
    return f'''
    <h1>Camera Required Input</h1>
    <ul>
        <li>Image String: {image}</li>
        <li>Signal 1: {s1}</li>
        <li>Signal 2: {s2}</li>
        <li>Signal 3: {s3}</li>
        <li>Signal 4: {s4}</li>
    </ul>
    '''

if __name__ == '__main__':
    app.run(debug=True)
