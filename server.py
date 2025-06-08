import os, queue, threading, datetime, math, wave, struct
import numpy as np, tensorflow as tf, joblib
from flask import (Flask, jsonify, request, Response,
                   stream_with_context, render_template_string,
                   send_from_directory)
from flask_cors import CORS

#  檔案 / SMTP 設定 
MODEL_PATH  = "improved_health_model.h5"
SCALER_PATH = "improved_health_scaler.pkl"
ALERT_WAV   = "static/alert.wav"

SMTP_USER     = "daniel40906@gmail.com"
SMTP_PASSWORD = "hbwnovitnzbcwanf"
SMTP_TO       = "C111141219@nkust.edu.tw"
SMTP_SERVER   = "smtp.gmail.com"
SMTP_PORT     = 465


FEATURE_ORDER = [
    "heart_rate", "spo2", "temperature",
    "age", "gender", "bmi", "hrv"
]

# 載入模型 AI 
try:
    ai_model  = tf.keras.models.load_model(MODEL_PATH)
    ai_scaler = joblib.load(SCALER_PATH)
    print("✅ AI 模型與 Scaler 載入成功")
except Exception as e:
    print("❌ AI 載入失敗：", e)
    ai_model = ai_scaler = None

#  Flask 
app = Flask(__name__)
CORS(app)
event_q = queue.Queue()          # SSE 佇列

#  首頁 (行動網頁)
INDEX_HTML = """
<!DOCTYPE html><meta charset="utf-8">
<title>跌倒監聽</title>
<style>body{font-family:sans-serif;text-align:center;padding:40px}
#msg{font-size:1.5rem;color:#555;margin:24px 0}
button{padding:12px 28px;font-size:1.2rem;border-radius:8px;border:none;background:#4caf50;color:#fff}
</style>
<h2>📡 跌倒事件監聽</h2>
<p id="msg">請先點擊下方按鈕啟用聲音</p>
<button id="btn">開始監聽</button>
<script>
let es=null;
btn.onclick=()=>{
  speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  es=new EventSource("/stream");
  es.onmessage=e=>{if(e.data==="fall") announce();};
  msg.textContent="已啟用，等待事件…"; btn.style.display="none";
};
function announce(){
  speechSynthesis.speak(new SpeechSynthesisUtterance("警告！警告！有人跌倒！"));
  new Audio("/static/alert.wav").play().catch(()=>{});
}
</script>"""
@app.route("/")          
def index(): return render_template_string(INDEX_HTML)

# SSE
@app.route("/stream")
def stream():
    def gen():
        while True: yield f"data: {event_q.get()}\\n\\n"
    return Response(stream_with_context(gen()),mimetype="text/event-stream")

# 跌倒回報
@app.route("/fall", methods=["POST"])
def fall():
    if bool(request.get_json(force=True).get("fall")):
        event_q.put("fall")
        threading.Thread(target=send_email, daemon=True).start()
        tts_power_shell()
    return jsonify({"status":"ok"})

# AI 評估 
def build_feature(p):
    age=float(p['age']); gender=1.0 if p['sex']=='F' else 0.0
    h=float(p['height'])/100; w=float(p['weight'])
    bmi=w/(h*h)
    vec=np.array([[ float(p['heartRate']), float(p['spo2']), float(p['bodyTemp']),
                    age, gender, bmi,
                    float(p.get('hrv', max(10,50-(age-25)*0.4-abs(float(p['heartRate'])-70)*0.2))) ]],
                 dtype=np.float32)
    return bmi, ai_scaler.transform(vec) if ai_scaler else vec

def advice_from_risk(r):
    ov=max(0,min(r["overall_risk"],1))
    high=max(r["cardiovascular_risk"],r["respiratory_risk"],r["metabolic_risk"],r["thermal_risk"])
    if ov>=0.8: return {"urgency":"emergency","recommend":"🚨 立即就醫","reason":[f"overall={ov:.2f}"]}
    if ov>=0.6: return {"urgency":"urgent","recommend":"⚠️ 24–48 小時內就醫","reason":[f"overall={ov:.2f}"]}
    if ov>=0.4 or high>0.5:
        return {"urgency":"moderate","recommend":"💡 1–2 週內安排檢查",
                "reason":[f"overall={ov:.2f}" if ov>=0.4 else f"單項={high:.2f}"]}
    return {"urgency":"low","recommend":"✅ 目前狀況良好","reason":[]}

@app.route("/eval", methods=["GET","POST"])
def eval_health():
    if ai_model is None: return jsonify({"error":"model not loaded"}),500

    # 先拿到 JSON
    payload = request.get_json(force=True)

    # --- 印出來看 ---
    print("===== 收到的 payload =====")
    print(payload)
    print("=========================")
    payload=request.get_json(force=True)
    bmi, feat = build_feature(payload)
    pred=ai_model.predict(feat,verbose=0)[0]
    risks={"cardiovascular_risk":float(pred[0]),"respiratory_risk":float(pred[1]),
           "metabolic_risk":float(pred[2]),"thermal_risk":float(pred[3]),
           "overall_risk":float(pred[4])}
    advice=advice_from_risk(risks)
    
    return jsonify({"bmi":round(bmi,1),"risks":risks,"advice":advice})

# —— 靜態音檔
@app.route("/static/<path:p>")
def static_file(p): return send_from_directory("static", p)

# 工具函式 
def tts_power_shell():
    try:
        os.system(
          "powershell -NoLogo -Command "
          "\"Add-Type -AssemblyName System.Speech;"
          "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
          "$s.Volume=100;1..3|%{$s.Speak('警告！警告！有人跌倒！')}\"")
    except: pass

def send_email():
    import smtplib; from email.mime.text import MIMEText; from email.header import Header
    msg=MIMEText("⚠️ 系統偵測到跌倒事件！請立即確認。","plain","utf-8")
    msg['Subject']=Header("⚠️ 跌倒警報","utf-8"); msg['From']=SMTP_USER; msg['To']=SMTP_TO
    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as s:
            s.login(SMTP_USER,SMTP_PASSWORD); s.sendmail(SMTP_USER,[SMTP_TO],msg.as_string())
        print("✅ Email 已寄出")
    except Exception as e: print("❌ Email 失敗：",e)

def ensure_alert():
    os.makedirs("static",exist_ok=True)
    if os.path.exists(ALERT_WAV): return
    with wave.open(ALERT_WAV,"w") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
        for i in range(22050):
            wf.writeframes(struct.pack('<h',int(16000*math.sin(2*math.pi*1000*i/22050))))
    print("⚠️ 已產生占位警報音 static/alert.wav，建議換成自訂聲音")

# 執行
if __name__=="__main__":
    ensure_alert()
    app.run(host="0.0.0.0", port=5000, threaded=True)
