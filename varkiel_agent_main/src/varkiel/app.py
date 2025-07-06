# SPDX-License-Identifier: MIT
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
# Copyright (c) 2025 ochoaughini. See LICENSE for full terms.
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import numpy as np
import os
import sqlite3
from central_controller import CentralController
from prometheus_client import start_http_server, Summary, Gauge

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# Simple user database
users = {'admin': {'password': 'varkiel123'}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Database setup
def init_db():
    conn = sqlite3.connect('varkiel.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY, user_id TEXT, input TEXT, output BLOB, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

# Start metrics server
start_http_server(8000)

# Create metrics
CACHE_HIT_RATIO = Gauge('meta_constraint_cache_hit_ratio', 'Cache hit ratio for MetaConstraintTree')
CACHE_SIZE = Gauge('meta_constraint_cache_size', 'Current cache size in bytes')

# Controllers
controller = CentralController()

# Routes
@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if users.get(username) and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/admin')
@login_required
def admin():
    if current_user.id != 'admin':
        return "Unauthorized", 403
    
    conn = sqlite3.connect('varkiel.db')
    c = conn.cursor()
    c.execute("SELECT * FROM sessions ORDER BY timestamp DESC LIMIT 50")
    sessions = c.fetchall()
    conn.close()
    
    return render_template('admin.html', sessions=sessions)

@app.route('/export/csv')
@login_required
def export_csv():
    # Simplified CSV export
    conn = sqlite3.connect('varkiel.db')
    c = conn.cursor()
    c.execute("SELECT * FROM sessions")
    sessions = c.fetchall()
    conn.close()
    
    csv = "id,user_id,input,timestamp\n"
    for session in sessions:
        csv += f"{session[0]},{session[1]},{session[2]},{session[4]}\n"
    
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=varkiel_sessions.csv"}
    )

@app.route('/process', methods=['POST'])
@login_required
def process_input():
    data = request.json
    
    # Convert input to vector
    input_text = data.get('input', '')
    input_vector = np.random.randn(256) * len(input_text) / 10.0
    
    # Process through Varkiel architecture
    output = controller.process_input(input_vector)
    
    # Store session
    conn = sqlite3.connect('varkiel.db')
    c = conn.cursor()
    c.execute("INSERT INTO sessions (user_id, input, output) VALUES (?, ?, ?)",
              (current_user.id, input_text, str(output.tolist())))
    conn.commit()
    conn.close()
    
    return jsonify({
        'output_vector': output.tolist(),
        'min_val': float(np.min(output)),
        'max_val': float(np.max(output)),
        'norm': float(np.linalg.norm(output))
    })

# New philosophical endpoints
@app.route('/varkiel/utter', methods=['POST'])
@login_required
def process_utterance():
    data = request.json
    input_text = data.get('input', '')
    
    # Convert text to vector (placeholder implementation)
    input_vector = np.random.randn(256)  # Replace with actual text embedding
    
    # Process through Varkiel's core architecture
    output_vector = controller.process_input(input_vector)
    
    # Convert vector to text (placeholder implementation)
    output_text = f"Processed vector: {output_vector.tolist()[:5]}..."
    
    # Store session
    conn = sqlite3.connect('varkiel.db')
    c = conn.cursor()
    c.execute("INSERT INTO sessions (user_id, input, output) VALUES (?, ?, ?)",
              (current_user.id, input_text, str(output_vector.tolist())))
    conn.commit()
    conn.close()
    
    return jsonify({
        'output': output_text,
        'reflection_phase': controller.last_reflection_phase
    })

@app.route('/varkiel/state', methods=['GET'])
@login_required
def get_state():
    return jsonify({
        'structural_state': controller.structural_engine.get_state(),
        'symbolic_state': controller.symbolic_engine.get_state(),
        'phenomenological_state': controller.phenomenological_tracker.get_state()
    })

@app.route('/varkiel/meta', methods=['GET'])
@login_required
def get_metadata():
    return jsonify(controller.get_configuration_metadata())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
