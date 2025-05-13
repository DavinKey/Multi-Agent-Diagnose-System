from flask import Flask, request, jsonify, render_template
from multi_agent import run_debate

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index_1.html")

@app.route('/run', methods=['POST'])
def run():
    data = request.get_json()
    case_text = data.get("case_text", "")
    print("✅ Received case_text:", case_text[:100])  
    result = run_debate(case_text)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)