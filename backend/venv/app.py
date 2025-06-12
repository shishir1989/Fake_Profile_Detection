from flask import Flask, request, jsonify
from flask_cors import CORS
from model_service import analyze_profile

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

@app.route('/api/analyze', methods=['POST'])
def analyze():
    username = request.form.get('username')
    bio = request.form.get('bio')
    profile_pic = request.files.get('profile-pic')

    # Dummy model analysis (replace with real ML logic later)
    result = analyze_profile(username, bio, profile_pic)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)