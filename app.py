from flask import Flask, render_template, request, jsonify
import analyzer   # connects to analyzer.py

app = Flask(__name__)

# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Optimize button endpoint
@app.route('/optimize', methods=['POST'])
def optimize():
    data = request.json

    # Inputs from UI
    target = data.get('target')
    num_stems = data.get('numStems')
    basis = data.get('basis')
    rename = data.get('rename')

    print(f"🚀 Received Request: Target={target}, Stems={num_stems}, Basis={basis}, Rename={rename}")

    # Run your analyzer logic
    analyzer.run_optimization(target, num_stems, basis, rename)

    return jsonify({
        "status": "success",
        "message": "Full DAW organisation complete!"
    })

@app.route('/export', methods=['POST'])
def export():

    data = request.json

    target = data.get('target')
    num_stems = data.get('numStems')
    basis = data.get('basis')
    rename = data.get('rename')

    print("📦 Exporting tracks...")

    analyzer.export_tracks(target, num_stems, basis)

    return jsonify({"status": "success", "message": "Export complete!"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)