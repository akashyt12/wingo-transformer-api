"""
WinGo Transformer v5 - 10 Feature + Transformer Ensemble
Rolling buffer + Live prediction
"""
import os
import json
import math
import time
import urllib.request
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# GAME DATA - BRIGHT-HOST-SPOT API
# ============================================================

GAME_API_BASE = "https://bright-host-spot.lovable.app/api/public"
GAME_ENDPOINTS = {
    "30s": "WinGo_30S",
    "1m":  "WinGo_1M",
    "3m":  "WinGo_3M",
    "5m":  "WinGo_5M",
}

class GameDataBuffer:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.max_size = 25

    def fetch_latest(self, game_key="30s"):
        endpoint = GAME_ENDPOINTS.get(game_key, "WinGo_30S")
        url = f"{GAME_API_BASE}/{endpoint}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = data.get("data", {}).get("list", [])
            results = []
            for item in items:
                num = item.get("number", "")
                issue = item.get("issueNumber", "")
                if str(num).isdigit() and issue:
                    results.append({"issue": str(issue), "number": int(num)})
            results.reverse()
            return results
        except Exception as e:
            print(f"  [FETCH ERROR] {e}")
            return []

    def update(self, game_key="30s"):
        with self.lock:
            new_data = self.fetch_latest(game_key)
            if not new_data:
                return
            existing = self.cache.get(game_key, [])
            existing_issues = {r["issue"] for r in existing}
            added = 0
            for item in new_data:
                if item["issue"] not in existing_issues:
                    existing.append(item)
                    existing_issues.add(item["issue"])
                    added += 1
            existing.sort(key=lambda x: x["issue"])
            if len(existing) > self.max_size:
                existing = existing[-self.max_size:]
            self.cache[game_key] = existing
            print(f"  [{game_key}] Cache: {len(existing)} records, +{added} new")

    def get_numbers(self, game_key="30s", count=20):
        with self.lock:
            data = self.cache.get(game_key, [])
            nums = [r["number"] for r in data]
            issues = [r["issue"] for r in data]
            return nums[-count:], issues[-count:]

    def get_numbers_display(self, game_key="30s", count=20):
        with self.lock:
            data = self.cache.get(game_key, [])
            nums = [r["number"] for r in data][::-1]
            issues = [r["issue"] for r in data][::-1]
            return nums[:count], issues[:count]

    def init_cache(self):
        for game in GAME_ENDPOINTS:
            self.update(game)
            time.sleep(0.2)

buffer = GameDataBuffer()

# ============================================================
# MODEL v2 ARCHITECTURE (original transformer)
# ============================================================

SEQUENCE_LEN = 20
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.1
DEVICE = "cpu"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class WinGoTransformer(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, num_layers=2, dropout=0.1, seq_len=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.number_embed = nn.Embedding(11, embed_dim, padding_idx=0)
        self.digit_embed = nn.Embedding(10, embed_dim // 4)
        self.dig_proj = nn.Linear(embed_dim // 4 * 3, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=seq_len + 10)
        self.volatility_proj = nn.Linear(3, embed_dim // 4)
        self.combine_proj = nn.Linear(embed_dim + embed_dim // 4, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head_bs = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, x):
        emb = self.number_embed(x)
        tens = x.unsqueeze(-1).expand(-1, -1, 3)
        ones = tens % 10
        tens = tens // 10
        digs = torch.cat([self.digit_embed(ones[:,:,0]),
                          self.digit_embed(ones[:,:,1]),
                          self.digit_embed(ones[:,:,2])], dim=-1)
        feat_list = []
        for i in range(x.size(1)):
            win = x[:, max(0, i-9):i+1]
            mean = win.float().mean(dim=1, keepdim=True)
            std = win.float().std(dim=1, keepdim=True, unbiased=False).clamp(min=0.1)
            last3 = win[:, -3:].float().mean(dim=1, keepdim=True) if win.size(1) >= 3 else mean
            feat_list.append(torch.cat([mean, std, last3], dim=1))
        vol_feats = torch.stack(feat_list, dim=1)
        vol_proj = self.volatility_proj(vol_feats)
        combined = torch.cat([emb + self.dig_proj(digs), vol_proj], dim=-1)
        x = self.combine_proj(combined)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.norm(x[:, -1, :])
        return self.head_bs(x)

# ============================================================
# 10-FEATURE MODEL (NEW)
# ============================================================

class TenFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        return self.net(x)

def extract_10_features(numbers, idx, seq_len=10):
    if idx < seq_len:
        return None
    w = numbers[idx - seq_len:idx]
    sum2 = sum(w[-2:])
    sum3 = sum(w[-3:])
    sum4 = sum(w[-4:])
    sum5 = sum(w[-5:])
    sum6 = sum(w[-6:])
    sum7 = sum(w[-7:])
    s2o = sum2 % 2 == 1
    s3o = sum3 % 2 == 1
    if s2o and s3o:
        f4 = 1
    elif not s2o and not s3o:
        f4 = 0
    elif s2o and not s3o:
        f4 = 0
    else:
        f4 = 1
    streak = 0
    for i in range(len(w) - 1, -1, -1):
        if w[i] == w[-1]:
            streak += 1
        else:
            break
    return [sum2, sum3, sum4, sum5, sum6, sum7, f4, w[-1], abs(w[-1] - w[-2]), streak]

def get_suggested_numbers(feat):
    sum2 = feat[0]
    sum3 = feat[1]
    s2o = sum2 % 2 == 1
    s3o = sum3 % 2 == 1
    if s2o and s3o:
        return [6, 8, 0]
    elif not s2o and not s3o:
        return [2, 4, 5]
    elif s2o and not s3o:
        return [1, 3, 5]
    else:
        return [7, 9, 0]

# ============================================================
# ENSEMBLE PREDICTOR (Transformer + 10-Feature)
# ============================================================

class Predictor:
    def __init__(self):
        self.tf_models = {}
        self.feat_model = None
        self.ready = False

    def load_models(self):
        for game in ["1m", "30s"]:
            try:
                model = WinGoTransformer(
                    embed_dim=EMBED_DIM, num_heads=NUM_HEADS,
                    num_layers=NUM_LAYERS, dropout=DROPOUT, seq_len=SEQUENCE_LEN
                ).to(DEVICE)
                path = os.path.join(os.path.dirname(__file__), f"wingo_transformer_{game}.pt")
                model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
                model.eval()
                self.tf_models[game] = model
                print(f"  [OK] Transformer {game} loaded")
            except Exception as e:
                print(f"  [FAIL] Transformer {game}: {e}")

        try:
            self.feat_model = TenFeatureNet()
            path = os.path.join(os.path.dirname(__file__), "wingo_10feat_best.pt")
            self.feat_model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
            self.feat_model.eval()
            print("  [OK] 10-Feature model loaded")
        except Exception as e:
            print(f"  [FAIL] 10-Feature: {e}")

        self.ready = len(self.tf_models) > 0 or self.feat_model is not None

    def predict_with_numbers(self, numbers, game="30s"):
        if not self.ready:
            return {"error": "Models not loaded"}
        if len(numbers) < 10:
            return {"error": f"Need at least 10 numbers, got {len(numbers)}"}

        # --- Frequency analysis (last 20 numbers) ---
        recent = numbers[-20:]
        freq = {}
        for n in range(10):
            freq[n] = recent.count(n)

        # --- Weighted score: recent numbers matter more ---
        scores = {}
        for n in range(10):
            score = 0
            for i, num in enumerate(recent):
                if num == n:
                    score += (i + 1)  # more recent = higher weight
            scores[n] = score

        # --- Gap analysis: numbers that haven't appeared in a while ---
        gaps = {}
        for n in range(10):
            gap = 0
            for i in range(len(recent) - 1, -1, -1):
                if recent[i] == n:
                    break
                gap += 1
            gaps[n] = gap

        # --- Transformer BIG/SMALL hint ---
        model = self.tf_models.get(game, self.tf_models.get("1m"))
        tf_hint = None
        if model and len(numbers) >= SEQUENCE_LEN:
            seq = numbers[-SEQUENCE_LEN:]
            x = torch.tensor([seq], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                bs_logits = model(x)
            tf_hint = "BIG" if bs_logits.argmax(1).item() == 1 else "SMALL"

        # --- Combine scores ---
        final_scores = {}
        for n in range(10):
            s = scores[n] * 2 + gaps[n] * 1.5 + freq[n] * 3
            if tf_hint == "BIG" and n >= 5:
                s *= 1.3
            elif tf_hint == "SMALL" and n <= 4:
                s *= 1.3
            final_scores[n] = round(s, 2)

        # --- Pick top 5 numbers with BIG/SMALL majority ---
        sorted_nums = sorted(final_scores, key=final_scores.get, reverse=True)

        if tf_hint == "BIG":
            big_nums = [n for n in sorted_nums if n >= 5]
            small_nums = [n for n in sorted_nums if n <= 4]
            top5 = big_nums[:3] + small_nums[:2]
            if len(top5) < 5:
                top5 += [n for n in sorted_nums if n not in top5][:5 - len(top5)]
        elif tf_hint == "SMALL":
            small_nums = [n for n in sorted_nums if n <= 4]
            big_nums = [n for n in sorted_nums if n >= 5]
            top5 = small_nums[:3] + big_nums[:2]
            if len(top5) < 5:
                top5 += [n for n in sorted_nums if n not in top5][:5 - len(top5)]
        else:
            top5 = sorted_nums[:5]

        best_num = top5[0]
        max_score = max(final_scores.values())
        confidence = round((final_scores[best_num] / max(max_score, 1)) * 100, 1)
        confidence = min(confidence, 95.0)

        return {
            "prediction": str(best_num),
            "number": best_num,
            "suggested_numbers": top5,
            "confidence": confidence,
            "scores": final_scores,
            "tf_hint": tf_hint,
            "game": game,
            "sequence": numbers[-10:],
        }

    def auto_predict(self, game="30s"):
        buffer.update(game)
        nums_model, issues_model = buffer.get_numbers(game, count=20)
        nums_display, issues_display = buffer.get_numbers_display(game, count=10)
        if len(nums_model) < 10:
            return {"error": "Not enough data", "cached": len(nums_model)}
        result = self.predict_with_numbers(nums_model, game)
        result["source"] = "bright-host-spot_live"
        result["cached_records"] = len(nums_model)
        result["latest_period"] = issues_display[0] if issues_display else "unknown"
        result["recent_numbers"] = nums_display
        return result

predictor = Predictor()

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "name": "WinGo Transformer v5 - 10 Feature + Ensemble",
        "version": "5.0",
        "status": "ready" if predictor.ready else "loading",
        "buffer_size": len(buffer.cache.get("30s", [])),
        "models": {
            "transformer": list(predictor.tf_models.keys()),
            "ten_feature": predictor.feat_model is not None,
        },
        "endpoints": {
            "GET /": "This page",
            "GET /status": "Model + buffer status",
            "GET /predict/auto?game=30s": "Live predict (ensemble!)",
            "GET /predict?game=30s&numbers=1,2,3,...": "Predict with your numbers",
            "GET /buffer?game=30s": "View cached numbers",
            "GET /refresh?game=30s": "Force refresh buffer",
        }
    })

@app.route('/status')
def status():
    return jsonify({
        "ready": predictor.ready,
        "transformer_models": list(predictor.tf_models.keys()),
        "ten_feature_model": predictor.feat_model is not None,
        "version": "5.0",
        "buffer": {k: len(v) for k, v in buffer.cache.items()},
    })

@app.route('/buffer')
def show_buffer():
    game = request.args.get('game', '30s')
    nums, issues = buffer.get_numbers_display(game, count=25)
    return jsonify({
        "game": game,
        "count": len(nums),
        "numbers": nums,
        "issues": issues,
        "latest": nums[0] if nums else "none",
    })

@app.route('/refresh')
def refresh():
    game = request.args.get('game', '30s')
    buffer.update(game)
    nums, issues = buffer.get_numbers_display(game, count=25)
    return jsonify({"game": game, "count": len(nums), "latest": nums[0] if nums else "none"})

@app.route('/predict/auto')
def auto_predict():
    game = request.args.get('game', '30s')
    return jsonify(predictor.auto_predict(game))

@app.route('/predict')
def predict_get():
    game = request.args.get('game', '30s')
    nums_str = request.args.get('numbers', '')
    if not nums_str:
        return jsonify(predictor.auto_predict(game))
    try:
        numbers = [int(x.strip()) for x in nums_str.split(',') if x.strip().isdigit()]
    except:
        return jsonify({"error": "Invalid numbers"}), 400
    return jsonify(predictor.predict_with_numbers(numbers, game))

@app.route('/predict', methods=['POST'])
def predict_post():
    data = request.get_json() or {}
    numbers = data.get('numbers', [])
    game = data.get('game', '30s')
    if not numbers:
        return jsonify(predictor.auto_predict(game))
    return jsonify(predictor.predict_with_numbers(numbers, game))

def auto_refresh():
    while True:
        time.sleep(1)
        try:
            for game in GAME_ENDPOINTS:
                buffer.update(game)
        except Exception as e:
            print(f"  [AUTO REFRESH ERROR] {e}")

if __name__ == '__main__':
    print("=" * 50)
    print("  WinGo Transformer v5 - 10 Feature + Ensemble")
    print("=" * 50)
    predictor.load_models()
    print("  Initializing buffer...")
    buffer.init_cache()
    t = threading.Thread(target=auto_refresh, daemon=True)
    t.start()
    port = int(os.environ.get('PORT', 5000))
    print(f"  Port: {port} | Ready: {predictor.ready}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
