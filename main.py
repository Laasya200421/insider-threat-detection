import os
import json
import joblib
import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
import requests
from functools import wraps
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, g as flask_g

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = os.environ.get('SECRET_KEY', 'threatlens-dev-secret-key-2024')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_DIR  = os.path.join(BASE_DIR, 'insider_threat_v4_bundle')
MODELS_DIR  = os.path.join(BUNDLE_DIR, 'models')
DB_PATH     = os.path.join(BASE_DIR, 'threatlens.db')

# ── Gemini API Keys (failover) ────────────────────────────────────────────────
GEMINI_KEYS = [
    "AIzaSyAyoOi-FnvkD1mTLrjFifyEJz31dAoHtPI",
    "AIzaSyAroJAeARfPQLkRiFe4L5rfaISOYgkuLgw",
    "AIzaSyAI5MA8mhWK4yKY3LmAPSW5jWHkKLfY6kE",
    "AIzaSyDAFZwPbJKrK4m-MBUk4m-IP9APtOD2_PI",
]
GEMINI_MODEL = "gemini-1.5-flash"

# ── Feature Columns ───────────────────────────────────────────────────────────
FEATURE_COLS = [
    'email_count', 'email_after_hours', 'email_weekend', 'unique_recipients',
    'pct_after_hours_email', 'file_access_count', 'file_after_hours',
    'file_weekend', 'pct_after_hours_file', 'login_count', 'login_after_hours',
    'login_weekend', 'unique_pcs', 'pct_after_hours_login', 'usb_events',
    'usb_after_hours', 'usb_weekend', 'pct_after_hours_usb',
    'email_count_zscore', 'file_access_count_zscore', 'login_count_zscore',
    'usb_events_zscore', 'unique_recipients_zscore', 'unique_pcs_zscore',
    'behavioral_anomaly_score', 'graph_degree', 'graph_out_degree',
    'graph_anomaly_score', 'avg_bert_threat_prob'
]

ISO_FEATURE_COLS = [
    'email_count', 'email_after_hours', 'email_weekend', 'unique_recipients',
    'pct_after_hours_email', 'file_access_count', 'file_after_hours',
    'file_weekend', 'pct_after_hours_file', 'login_count', 'login_after_hours',
    'login_weekend', 'unique_pcs', 'pct_after_hours_login', 'usb_events',
    'usb_after_hours', 'usb_weekend', 'pct_after_hours_usb',
    'email_count_zscore', 'file_access_count_zscore', 'login_count_zscore',
    'usb_events_zscore', 'unique_recipients_zscore', 'unique_pcs_zscore',
]
ISO_INDICES = [FEATURE_COLS.index(c) for c in ISO_FEATURE_COLS]

W_XGB = 0.30; W_LR = 0.20; W_BERT = 0.20; W_GRAPH = 0.15; W_ISO = 0.15

iso_forest = None; xgb_clf = None; lr_clf = None; scaler = None
graph_data = None; results_df = None; processed_df = None; top10_df = None; metadata = {}


# ── DB Init ───────────────────────────────────────────────────────────────────
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        role TEXT DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT NOT NULL,
        risk_score REAL,
        alert TEXT,
        malicious_prob REAL,
        iso_score REAL,
        evidence TEXT,
        input_data TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    # Create default admin
    try:
        c.execute("INSERT OR IGNORE INTO users (email, password, first_name, last_name, role) VALUES (?,?,?,?,?)",
                  ('admin@threatlens.ai', 'Admin@123', 'Admin', 'User', 'admin'))
    except:
        pass
    conn.commit()
    conn.close()


def get_db():
    if 'db' not in flask_g.__dict__:
        flask_g.db = sqlite3.connect(DB_PATH)
        flask_g.db.row_factory = sqlite3.Row
    return flask_g.db


@app.teardown_appcontext
def close_db(e=None):
    db = flask_g.__dict__.pop('db', None)
    if db: db.close()


# ── Gemini API Call with Failover ─────────────────────────────────────────────
def call_gemini(prompt):
    models_to_try = [GEMINI_MODEL, "gemini-1.5-flash-latest", "gemini-pro"]
    for key in GEMINI_KEYS:
        for model in models_to_try:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                resp = requests.post(url, json=payload, timeout=45)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    return text
                elif resp.status_code == 404:
                    continue  # Model not found, try next model
                elif resp.status_code in [429, 500, 503]:
                    break  # Rate limit/server error on this key, try next key
            except Exception:
                continue
    return None


# ── Model Helpers ─────────────────────────────────────────────────────────────
def load_models():
    global iso_forest, xgb_clf, lr_clf, scaler, graph_data
    global results_df, processed_df, top10_df, metadata

    def mp(f): return os.path.join(MODELS_DIR, f)
    def bp(f): return os.path.join(BUNDLE_DIR, f)

    for attr, fname in [('iso_forest','isolation_forest_behavioral.pkl'),('xgb_clf','xgboost_threat.pkl'),('lr_clf','logistic_regression_threat.pkl'),('scaler','feature_scaler.pkl')]:
        full = mp(fname)
        if os.path.exists(full): globals()[attr] = joblib.load(full); print(f'Loaded: {fname}')
        else: print(f'WARNING: {fname} not found')

    gm = mp('graph_metrics.json')
    if os.path.exists(gm):
        with open(gm) as f: graph_data = json.load(f)

    for attr, fname in [('results_df','results.csv'),('processed_df','processed_features.csv'),('top10_df','top10_high_risk.csv')]:
        full = bp(fname)
        if os.path.exists(full): globals()[attr] = pd.read_csv(full); print(f'Loaded: {fname}')

    mj = bp('model_metadata.json')
    if os.path.exists(mj):
        with open(mj) as f: metadata = json.load(f)


def classify_risk(score):
    if score < 20: return 'Normal'
    if score < 40: return 'Low'
    if score < 60: return 'Medium'
    if score < 80: return 'High'
    return 'Critical'


def alert_color(level):
    return {'Normal':'#22c55e','Low':'#84cc16','Medium':'#f59e0b','High':'#f97316','Critical':'#ef4444'}.get(level,'#6b7280')


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('logged_in') or session.get('user_role') != 'admin':
            return redirect(url_for('login_page'))
        return f(*args, **kwargs)
    return decorated


# ── Auth Routes ───────────────────────────────────────────────────────────────
@app.route('/admin/signup')
def admin_signup_page():
    if session.get('logged_in') and session.get('user_role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_signup.html')


@app.route('/api/admin/signup', methods=['POST'])
def api_admin_signup():
    data = request.get_json()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    first_name = (data.get('first_name') or '').strip()
    last_name = (data.get('last_name') or '').strip()
    admin_code = (data.get('admin_code') or '').strip()
    if not email or not password or not first_name or not last_name:
        return jsonify({'success': False, 'error': 'All fields are required.'})
    if admin_code != 'THREATLENS-ADMIN-2024':
        return jsonify({'success': False, 'error': 'Invalid admin access code. Contact your system administrator.'})
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters.'})
    db = get_db()
    existing = db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
    if existing:
        return jsonify({'success': False, 'error': 'An account with this email already exists.'})
    db.execute('INSERT INTO users (email,password,first_name,last_name,role) VALUES (?,?,?,?,?)',
               (email, password, first_name, last_name, 'admin'))
    db.commit()
    session['logged_in'] = True
    session['user_email'] = email
    session['user_name'] = f'{first_name} {last_name}'
    session['user_role'] = 'admin'
    return jsonify({'success': True, 'redirect': '/admin'})


@app.route('/login')
def login_page():
    if session.get('logged_in'):
        return redirect(url_for('analyze') if session.get('user_role') != 'admin' else url_for('admin_dashboard'))
    return render_template('login.html')


@app.route('/signup')
def signup_page():
    if session.get('logged_in'): return redirect(url_for('analyze'))
    return render_template('signup.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password are required.'})
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    if not user or user['password'] != password:
        return jsonify({'success': False, 'error': 'Invalid email or password.'})
    session['logged_in'] = True
    session['user_email'] = email
    session['user_name'] = f"{user['first_name']} {user['last_name']}"
    session['user_role'] = user['role']
    redirect_url = '/admin' if user['role'] == 'admin' else '/analyze'
    return jsonify({'success': True, 'redirect': redirect_url})


@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    first_name = (data.get('first_name') or '').strip()
    last_name = (data.get('last_name') or '').strip()
    role = data.get('role') or 'user'
    if not email or not password or not first_name or not last_name:
        return jsonify({'success': False, 'error': 'All fields are required.'})
    if len(password) < 8:
        return jsonify({'success': False, 'error': 'Password must be at least 8 characters.'})
    db = get_db()
    existing = db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
    if existing:
        return jsonify({'success': False, 'error': 'An account with this email already exists.'})
    db.execute('INSERT INTO users (email,password,first_name,last_name,role) VALUES (?,?,?,?,?)',
               (email, password, first_name, last_name, 'user'))
    db.commit()
    session['logged_in'] = True
    session['user_email'] = email
    session['user_name'] = f"{first_name} {last_name}"
    session['user_role'] = 'user'
    return jsonify({'success': True, 'redirect': '/analyze'})


# ── Session User API (used by all pages for navbar user info) ─────────────────
@app.route('/api/session_user')
def api_session_user():
    if not session.get('logged_in'):
        return jsonify({'logged_in': False, 'name': '', 'email': '', 'role': ''})
    return jsonify({
        'logged_in': True,
        'name': session.get('user_name', ''),
        'email': session.get('user_email', ''),
        'role': session.get('user_role', 'user')
    })


# ── Page Routes ───────────────────────────────────────────────────────────────
@app.route('/')
def index(): return render_template('index.html')

@app.route('/dashboard')
@login_required
def dashboard(): return render_template('dashboard.html')

@app.route('/analyze')
@login_required
def analyze(): return render_template('analyze.html')

@app.route('/analysis')
@login_required
def analysis_page(): return render_template('analysis.html')

@app.route('/history')
@login_required
def history(): return render_template('history.html')

@app.route('/recommendation')
@login_required
def recommendation(): return render_template('recommendation.html')

@app.route('/admin')
@admin_required
def admin_dashboard(): return render_template('admin.html')


# ── Prediction API ────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if xgb_clf is None or scaler is None or iso_forest is None:
        return jsonify({'error': 'Models not loaded'}), 500
    try:
        data = request.get_json()
        row = [float(data.get(col, 0)) for col in FEATURE_COLS]
        X = np.array(row).reshape(1, -1)
        X_scaled = scaler.transform(X)
        xgb_prob = float(xgb_clf.predict_proba(X_scaled)[0][1])
        lr_prob = float(lr_clf.predict_proba(X_scaled)[0][1]) if lr_clf else xgb_prob
        X_iso = X_scaled[:, ISO_INDICES]
        iso_raw = iso_forest.score_samples(X_iso)[0]
        iso_score = float(np.clip(1.0 - ((iso_raw + 0.5) / 1.0), 0, 1))
        bert_prob = float(data.get('avg_bert_threat_prob', 0.075))
        graph_score = float(data.get('graph_anomaly_score', 0.0))
        risk_raw = (W_XGB * xgb_prob + W_LR * lr_prob + W_BERT * bert_prob + W_GRAPH * graph_score + W_ISO * iso_score)
        risk_score = round(float(np.clip(risk_raw, 0, 1) * 100), 2)
        alert = classify_risk(risk_score)

        evidence = []
        if data.get('email_after_hours', 0) > 50: evidence.append(f"{int(data['email_after_hours'])} after-hours emails detected")
        if data.get('usb_events', 0) > 10: evidence.append(f"{int(data['usb_events'])} USB events logged")
        if data.get('login_after_hours', 0) > 100: evidence.append(f"{int(data['login_after_hours'])} after-hours logins detected")
        if data.get('file_after_hours', 0) > 20: evidence.append(f"{int(data['file_after_hours'])} after-hours file accesses")
        if data.get('behavioral_anomaly_score', 0) > 0.5: evidence.append(f"High behavioral anomaly score: {round(data['behavioral_anomaly_score'],3)}")
        if graph_score > 0.7: evidence.append(f"Anomalous communication graph (score: {round(graph_score,3)})")
        if data.get('pct_after_hours_usb', 0) > 0.5: evidence.append('Majority of USB activity occurs off-hours')
        if not evidence: evidence.append('No significant anomalies detected')

        result = {
            'risk_score': risk_score, 'alert': alert, 'color': alert_color(alert),
            'malicious_prob': round(xgb_prob * 100, 1),
            'iso_score': round(iso_score * 100, 1),
            'evidence': evidence,
            'xgb_prob': round(xgb_prob * 100, 1),
            'lr_prob': round(lr_prob * 100, 1),
            'bert_prob': round(bert_prob * 100, 1),
            'graph_score': round(graph_score * 100, 1),
        }

        # Save to history if logged in
        if session.get('logged_in'):
            try:
                db = get_db()
                input_summary = f"Emails:{int(data.get('email_count',0))} | Logins:{int(data.get('login_count',0))} | USB:{int(data.get('usb_events',0))} | Files:{int(data.get('file_access_count',0))}"
                db.execute(
                    'INSERT INTO predictions (user_email,risk_score,alert,malicious_prob,iso_score,evidence,input_data) VALUES (?,?,?,?,?,?,?)',
                    (session['user_email'], risk_score, alert, round(xgb_prob*100,1), round(iso_score*100,1),
                     json.dumps(evidence), input_summary)
                )
                db.commit()
                last_id = db.execute('SELECT last_insert_rowid()').fetchone()[0]
                result['prediction_id'] = last_id
            except Exception as e:
                print(f"DB save error: {e}")

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ── History API ───────────────────────────────────────────────────────────────
@app.route('/api/history')
@login_required
def api_history():
    db = get_db()
    rows = db.execute(
        'SELECT * FROM predictions WHERE user_email=? ORDER BY created_at DESC LIMIT 50',
        (session['user_email'],)
    ).fetchall()
    result = []
    for r in rows:
        result.append({
            'id': r['id'], 'risk_score': r['risk_score'], 'alert': r['alert'],
            'malicious_prob': r['malicious_prob'], 'iso_score': r['iso_score'],
            'evidence': json.loads(r['evidence']) if r['evidence'] else [],
            'input_data': r['input_data'],
            'created_at': r['created_at']
        })
    return jsonify(result)


@app.route('/api/history/<int:pid>')
@login_required
def api_history_detail(pid):
    db = get_db()
    row = db.execute('SELECT * FROM predictions WHERE id=? AND user_email=?', (pid, session['user_email'])).fetchone()
    if not row: return jsonify({'error': 'Not found'}), 404
    return jsonify({
        'id': row['id'], 'risk_score': row['risk_score'], 'alert': row['alert'],
        'malicious_prob': row['malicious_prob'], 'iso_score': row['iso_score'],
        'evidence': json.loads(row['evidence']) if row['evidence'] else [],
        'input_data': row['input_data'], 'created_at': row['created_at']
    })


# ── Fallback Recommendation Generator ───────────────────────────────────────
def generate_fallback_recommendation(risk_score, alert, evidence, malicious_prob):
    score = float(risk_score)
    ev_text = ', '.join(evidence) if evidence else 'No specific anomalies'

    if alert == 'Critical':
        urgency = 'CRITICAL — Immediate action required within 1-2 hours.'
        imm = ['Immediately suspend the employee\'s network and system access pending investigation','Alert the CISO, HR, and Legal team simultaneously','Preserve all digital evidence — do NOT notify the employee yet','Engage your incident response team and begin a formal investigation','Isolate the employee\'s workstation and revoke VPN/cloud access']
        strategies = ['Deploy DLP (Data Loss Prevention) tools on all endpoints in the same department','Conduct a full audit of all data accessed, copied, or transferred in the last 30 days','Review email logs for exfiltration patterns (large attachments, personal email forwarding)','Implement emergency access reviews across all privileged accounts','Coordinate with legal for potential forensic investigation and chain-of-custody procedures','Engage law enforcement if evidence of data theft is confirmed']
        dos = ['Document every step of the investigation with timestamps','Follow HR and legal protocols before any confrontation','Preserve forensic integrity — use write-blockers when imaging drives','Maintain strict confidentiality — limit knowledge to need-to-know personnel']
        donts = ['Do NOT alert the employee or their direct manager prematurely','Do NOT delete or modify any logs or files','Do NOT conduct informal interrogations without HR and Legal present','Do NOT allow continued access while investigation is ongoing']
        monitoring = ['24/7 real-time monitoring of all remaining access for this user','Daily review of all authentication logs and badge access records','Automated alerting for any lateral movement attempts','Full packet capture on any remaining active sessions','Weekly risk reassessment until case is closed']
        compliance = 'This situation likely requires legal counsel involvement. Ensure compliance with GDPR, HIPAA, or applicable data protection regulations when handling employee data. Document the investigation under attorney-client privilege where possible. HR must be involved before any disciplinary action.'
        timeline = 'Reassess risk daily during active investigation. If no criminal activity is confirmed within 72 hours, move to High-risk monitoring protocol. Full case review with legal team within 7 days.'
    elif alert == 'High':
        urgency = 'HIGH — Escalation required within 24 hours.'
        imm = ['Notify the security team lead and direct manager through secure channels','Schedule an access review for this employee\'s account within 4 hours','Enable enhanced logging on all systems this user accesses','Review the last 14 days of activity for anomalous patterns','Prepare an incident report with evidence flags for security leadership']
        strategies = ['Implement step-up authentication (MFA) for all sensitive system access','Restrict USB and external storage device usage via policy enforcement','Schedule a confidential HR conversation about acceptable use policies','Apply principle of least privilege — remove any excessive access rights','Enable email scanning for large attachments or unusual recipient patterns','Set up automated alerts for after-hours system access']
        dos = ['Increase monitoring frequency to real-time for this user','Engage HR proactively to document concerns through proper channels','Back up critical systems and data this user has access to','Review and tighten access controls across the department']
        donts = ['Do NOT share investigation details with colleagues of the employee','Do NOT take punitive action without documented evidence and HR approval','Do NOT ignore the signals — high risk profiles escalate rapidly','Do NOT grant any additional access or privileges during this period']
        monitoring = ['Enable real-time alerting for this user across all systems','Daily review of USB, file transfer, and email activity','Weekly risk score reassessment using the analyzer','Monitor for after-hours logins and off-network VPN usage','Alert on any attempts to access HR, payroll, or sensitive databases']
        compliance = 'Ensure all monitoring activities comply with your organization\'s acceptable use policy and applicable labor laws. In most jurisdictions, employee monitoring on company systems is permitted with proper disclosure. Consult HR and legal before escalating to formal investigation.'
        timeline = 'Reassess in 7 days. If score remains High or escalates, initiate formal investigation. If score drops to Medium, continue standard enhanced monitoring for 30 days.'
    elif alert == 'Medium':
        urgency = 'MEDIUM — Review within 48-72 hours.'
        imm = ['Flag the user account for enhanced monitoring in your SIEM','Notify the direct manager through appropriate channels for awareness','Review recent access logs for the past 7 days','Check if any policy violations are associated with the flagged behaviors','Document the current risk profile for future comparison']
        strategies = ['Implement periodic access reviews for this employee (bi-weekly)','Reinforce security awareness training, especially around USB and after-hours policies','Review and right-size the user\'s access permissions','Set up behavioral baseline alerts for deviation detection','Encourage use of official data transfer channels vs. USB/personal email']
        dos = ['Maintain documentation of all anomalous behaviors observed','Use this as an opportunity to refresh security training for the team','Review access rights quarterly for all users in this risk band','Keep an open line of communication with HR for any HR-related concerns']
        donts = ['Do NOT overreact — medium risk requires monitoring, not immediate action','Do NOT share monitoring status with the employee or peers','Do NOT ignore the trend — medium risk can escalate without intervention']
        monitoring = ['Weekly review of activity logs','Monthly risk score reassessment','Alert on score increases above 60 (High threshold)','Monitor USB plug events and after-hours login frequency','Track any requests for elevated access or permission changes']
        compliance = 'Medium risk situations typically do not require legal involvement yet. However, ensure monitoring is conducted in line with your employee privacy policy. Document all observations through proper HR channels in case escalation becomes necessary.'
        timeline = 'Reassess in 30 days. If score increases to High, escalate immediately. If score drops to Low or Normal, reduce monitoring frequency to monthly.'
    else:
        urgency = 'LOW/NORMAL — Routine monitoring recommended.'
        imm = ['No immediate action required','Note the assessment in the employee monitoring log','Schedule routine access review as per standard policy','Ensure standard security training is up to date for this user']
        strategies = ['Maintain standard monitoring protocols','Ensure periodic access reviews (quarterly)','Keep security awareness training current','Document current baseline for future comparison']
        dos = ['Continue standard monitoring practices','Maintain current access controls','Keep documentation current']
        donts = ['Do NOT relax monitoring entirely based on low score alone','Do NOT skip quarterly access reviews']
        monitoring = ['Monthly routine log review','Quarterly risk reassessment','Standard alert thresholds apply']
        compliance = 'Standard compliance monitoring is appropriate. Ensure routine access reviews and security training are maintained per organizational policy.'
        timeline = 'Reassess in 90 days as part of standard quarterly review cycle.'

    report = f"""## 1. Executive Summary

This employee has been flagged with a **{alert} risk level** (score: {score}/100) by the ThreatLens 5-model AI ensemble. The primary evidence signals include: {ev_text}. {urgency} The malicious probability score from XGBoost is **{malicious_prob}%**, indicating {'a high likelihood' if float(malicious_prob) > 60 else 'an elevated possibility' if float(malicious_prob) > 30 else 'a lower but notable possibility'} of malicious insider activity.

## 2. Immediate Actions Required

- {chr(10)+'- '.join(imm)}

## 3. Risk Mitigation Strategies

- {chr(10)+'- '.join(strategies)}

## 4. Do's and Don'ts

### Do's
- {chr(10)+'- '.join(dos)}

### Don'ts
- {chr(10)+'- '.join(donts)}

## 5. Monitoring Recommendations

- {chr(10)+'- '.join(monitoring)}

## 6. Compliance & Legal Considerations

{compliance}

## 7. Risk Reassessment Timeline

{timeline}

---
*⚠️ Note: This recommendation was generated by ThreatLens fallback engine (Gemini AI unavailable). All recommendations are based on CERT r4.2 insider threat research and industry best practices.*"""
    return report


# ── Recommendation API (Gemini) ───────────────────────────────────────────────
@app.route('/api/recommendation', methods=['POST'])
@login_required
def api_recommendation():
    data = request.get_json()
    risk_score = data.get('risk_score', 0)
    alert = data.get('alert', 'Normal')
    evidence = data.get('evidence', [])
    malicious_prob = data.get('malicious_prob', 0)

    prompt = f"""You are a senior cybersecurity expert specializing in insider threat mitigation.

An employee has been flagged by an AI threat detection system with the following profile:
- Risk Score: {risk_score}/100
- Risk Level: {alert}
- Malicious Probability: {malicious_prob}%
- Evidence Flags: {', '.join(evidence)}

Provide a structured, professional security recommendation report with the following exact sections:

## 1. Executive Summary
A 2-3 sentence summary of the risk situation.

## 2. Immediate Actions Required
List 4-5 specific actions the security team should take NOW (within 24 hours).

## 3. Risk Mitigation Strategies
List 5-6 medium-term strategies to reduce this specific risk profile.

## 4. Do's and Don'ts
### Do's (list 4-5 items)
### Don'ts (list 4-5 items)

## 5. Monitoring Recommendations
List 4-5 specific monitoring activities to track this employee.

## 6. Compliance & Legal Considerations
2-3 sentences about HR, legal, and compliance aspects.

## 7. Risk Reassessment Timeline
When and how to reassess this employee's risk.

Be specific, professional, and actionable. Use the evidence flags to tailor your recommendations."""

    result = call_gemini(prompt)
    if result:
        return jsonify({'success': True, 'recommendation': result, 'source': 'gemini'})
    else:
        fallback = generate_fallback_recommendation(risk_score, alert, evidence, malicious_prob)
        return jsonify({'success': True, 'recommendation': fallback, 'source': 'fallback'})


# ── Notifications API ────────────────────────────────────────────────────────
@app.route('/api/notifications')
@login_required
def api_notifications():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM predictions WHERE user_email=? AND (alert='Critical' OR alert='High') ORDER BY created_at DESC LIMIT 10",
        (session['user_email'],)
    ).fetchall()
    notifs = []
    for r in rows:
        notifs.append({
            'id': r['id'],
            'alert': r['alert'],
            'risk_score': r['risk_score'],
            'input_data': r['input_data'],
            'created_at': r['created_at']
        })
    return jsonify(notifs)


# ── Admin APIs ────────────────────────────────────────────────────────────────
@app.route('/api/admin/users')
@admin_required
def api_admin_users():
    db = get_db()
    users = db.execute('SELECT id,email,first_name,last_name,role,created_at FROM users ORDER BY created_at DESC').fetchall()
    return jsonify([dict(u) for u in users])


@app.route('/api/admin/predictions')
@admin_required
def api_admin_predictions():
    db = get_db()
    rows = db.execute('SELECT * FROM predictions ORDER BY created_at DESC LIMIT 500').fetchall()
    result = []
    for r in rows:
        result.append({
            'id': r['id'], 'user_email': r['user_email'], 'risk_score': r['risk_score'],
            'alert': r['alert'], 'malicious_prob': r['malicious_prob'],
            'iso_score': r['iso_score'],
            'input_data': r['input_data'], 'created_at': r['created_at']
        })
    return jsonify(result)


@app.route('/api/admin/stats')
@admin_required
def api_admin_stats():
    db = get_db()
    total_users = db.execute('SELECT COUNT(*) FROM users WHERE role != "admin"').fetchone()[0]
    total_preds = db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    critical_preds = db.execute('SELECT COUNT(*) FROM predictions WHERE alert="Critical"').fetchone()[0]
    high_preds = db.execute('SELECT COUNT(*) FROM predictions WHERE alert="High"').fetchone()[0]
    avg_row = db.execute('SELECT AVG(risk_score) FROM predictions').fetchone()
    avg_risk = round(float(avg_row[0]), 1) if avg_row and avg_row[0] is not None else 0
    return jsonify({
        'total_users': total_users, 'total_predictions': total_preds,
        'critical_predictions': critical_preds, 'high_predictions': high_preds,
        'avg_risk_score': avg_risk
    })


# ── Original Dashboard APIs ────────────────────────────────────────────────────
@app.route('/api/stats')
def api_stats():
    if results_df is None: return jsonify({'error': 'Models not loaded'}), 500
    df = results_df.copy()
    total = len(df); critical = int((df['risk_score'] >= 80).sum())
    high = int(((df['risk_score'] >= 60) & (df['risk_score'] < 80)).sum())
    medium = int(((df['risk_score'] >= 40) & (df['risk_score'] < 60)).sum())
    xgb_auc = round(float(metadata.get('xgb_val_auc', 0.998)) * 100, 2)
    lr_auc = round(float(metadata.get('lr_cv_auc', 0.998)) * 100, 2)
    return jsonify({'total_employees': total, 'critical_count': critical, 'high_count': high,
                    'medium_count': medium, 'avg_risk_score': round(float(df['risk_score'].mean()), 2),
                    'max_risk_score': round(float(df['risk_score'].max()), 2),
                    'model_accuracy': round((xgb_auc + lr_auc) / 2, 2), 'model_auc': xgb_auc})


@app.route('/api/top_employees')
def api_top_employees():
    src = top10_df if top10_df is not None else results_df
    if src is None: return jsonify([])
    df = src.copy().sort_values('risk_score', ascending=False).head(10)
    results = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        score = float(row['risk_score']); alert = str(row.get('alert_classification', classify_risk(score)))
        results.append({'rank': rank, 'name': str(row.get('user', f'EMP-{rank:03d}')),
                        'risk_score': round(score, 1), 'alert': alert, 'color': alert_color(alert),
                        'xgb_prob': round(float(row.get('xgb_threat_prob', 0)) * 100, 1),
                        'bert_prob': round(float(row.get('avg_bert_threat_prob', 0)) * 100, 1),
                        'graph_anomaly': round(float(row.get('graph_anomaly_score', 0)), 3),
                        'behav_score': round(float(row.get('behavioral_anomaly_score', 0)), 3),
                        'evidence': str(row.get('evidence_summary', 'N/A'))})
    return jsonify(results)


@app.route('/api/risk_distribution')
def api_risk_distribution():
    src = results_df if results_df is not None else processed_df
    if src is None: return jsonify({})
    df = src.copy()
    if 'alert_classification' not in df.columns: df['alert_classification'] = df['risk_score'].apply(classify_risk)
    counts = df['alert_classification'].value_counts().to_dict()
    order = ['Normal', 'Low', 'Medium', 'High', 'Critical']
    return jsonify({'labels': order, 'values': [counts.get(k, 0) for k in order], 'colors': [alert_color(k) for k in order]})


@app.route('/api/risk_timeline')
def api_risk_timeline():
    src = results_df if results_df is not None else processed_df
    if src is None: return jsonify({})
    df = src.copy()
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['Normal (0-20)', 'Low (20-40)', 'Medium (40-60)', 'High (60-80)', 'Critical (80-100)']
    counts = pd.cut(df['risk_score'], bins=bins, labels=labels, include_lowest=True).value_counts()
    return jsonify({'labels': labels, 'values': [int(counts.get(l, 0)) for l in labels],
                    'colors': ['#22c55e', '#84cc16', '#f59e0b', '#f97316', '#ef4444']})


@app.route('/api/feature_importance')
def api_feature_importance():
    if xgb_clf is None: return jsonify({})
    try:
        importances = xgb_clf.feature_importances_
        paired = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)[:10]
        total = sum(p[1] for p in paired) or 1
        return jsonify({'labels': [p[0].replace('_', ' ').title() for p in paired],
                        'values': [round(float(p[1] / total) * 100, 2) for p in paired]})
    except Exception as e: return jsonify({'error': str(e)})


@app.route('/api/isolation_scores')
def api_isolation_scores():
    src = results_df if results_df is not None else processed_df
    if src is None: return jsonify({})
    df = src.copy()
    score_col = next((c for c in ['behavioral_anomaly_score', 'n_behav', 'n_iso'] if c in df.columns), None)
    if score_col is None and 'risk_score' in df.columns:
        df['_ano'] = df['risk_score'] / 100.0; score_col = '_ano'
    if score_col is None: return jsonify({'error': 'No anomaly score column found'})
    scores = df[score_col].dropna().astype(float)
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ['0.0-0.1','0.1-0.2','0.2-0.3','0.3-0.4','0.4-0.5','0.5-0.6','0.6-0.7','0.7-0.8','0.8-0.9','0.9-1.0']
    counts = pd.cut(scores, bins=bins, labels=labels, right=False).value_counts().reindex(labels, fill_value=0)
    anomaly_count = int((df['iso_outlier'] == 1).sum()) if 'iso_outlier' in df.columns else int((scores > 0.5).sum())
    return jsonify({'labels': labels, 'counts': counts.tolist(), 'anomaly_count': anomaly_count,
                    'normal_count': len(df) - anomaly_count, 'total': len(df)})


@app.route('/api/graph_metrics')
def api_graph_metrics():
    src = processed_df if processed_df is not None else results_df
    if src is None: return jsonify({})
    df = src.copy()
    if 'alert_classification' not in df.columns: df['alert_classification'] = df['risk_score'].apply(classify_risk)
    grp_col = 'alert_classification'
    G = nx.DiGraph()
    groups = df[grp_col].dropna().unique().tolist()
    for g in groups: G.add_node(g)
    agg = df.groupby(grp_col).agg(avg_risk=('risk_score','mean'), count=('risk_score','count')).reset_index()
    for i, src_row in agg.iterrows():
        for j, tgt_row in agg.iterrows():
            if i == j: continue
            w = float(src_row['count'] * tgt_row['count']) / max(float(len(df)), 1)
            if w > 0: G.add_edge(str(src_row[grp_col]), str(tgt_row[grp_col]), weight=w)
    pagerank = nx.pagerank(G, alpha=0.85, weight='weight')
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)
    in_deg = dict(G.in_degree(weight='weight')); out_deg = dict(G.out_degree(weight='weight'))
    degree = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes}
    avg_risk_map = dict(zip(agg[grp_col], agg['avg_risk']))
    max_risk = max(avg_risk_map.values()) or 1; max_bt = max(betweenness.values()) or 1
    graph_anomaly = {n: round((betweenness[n]/max_bt)*0.5 + (avg_risk_map.get(n,0)/max_risk)*0.5, 6) for n in G.nodes}
    sorted_nodes = sorted(pagerank.keys(), key=lambda n: pagerank[n], reverse=True)
    return jsonify({'labels': sorted_nodes,
                    'graph_pagerank': [round(pagerank[n], 6) for n in sorted_nodes],
                    'graph_betweenness': [round(betweenness[n], 6) for n in sorted_nodes],
                    'graph_degree': [round(degree[n], 6) for n in sorted_nodes],
                    'graph_anomaly_score': [graph_anomaly[n] for n in sorted_nodes],
                    'pagerank_max': round(float(max(pagerank.values())), 6),
                    'betweenness_max': round(float(max(betweenness.values())), 6),
                    'graph_anomaly_max': round(float(max(graph_anomaly.values())), 6),
                    'high_graph_anomaly': int(sum(1 for v in graph_anomaly.values() if v > 0.5)),
                    'node_count': G.number_of_nodes(), 'edge_count': G.number_of_edges()})


@app.route('/api/department_breakdown')
def api_department_breakdown():
    src = results_df if results_df is not None else processed_df
    if src is None: return jsonify([])
    df = src.copy()
    if 'alert_classification' not in df.columns: df['alert_classification'] = df['risk_score'].apply(classify_risk)
    groups = df.groupby('alert_classification').agg(avg_risk=('risk_score','mean'), max_risk=('risk_score','max'), count=('risk_score','count')).reset_index()
    result = []
    for _, row in groups.iterrows():
        lbl = str(row['alert_classification'])
        result.append({'department': lbl, 'avg_risk': round(float(row['avg_risk']), 2),
                       'max_risk': round(float(row['max_risk']), 2), 'count': int(row['count']),
                       'alert': lbl, 'color': alert_color(lbl)})
    return jsonify(result)


if __name__ == '__main__':
    init_db()
    load_models()
    app.run(debug=True, port=5000)
