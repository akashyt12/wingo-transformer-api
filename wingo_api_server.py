"""
WinGo Predictor v9 - Markov + XGBoost + Random Forest
3 AI Models with Majority Voting - Number Prediction 0-9
"""
import os
import json
import time
import random
import urllib.request
import threading
import numpy as np
from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

# XGBoost
from xgboost import XGBClassifier

# Sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

# ============================================================
# GAME DATA - LIVE API
# ============================================================

GAME_API_BASE = "https://draw.ar-lottery01.com/WinGo"
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
        self.max_size = 200

    def fetch_latest(self, game_key="30s"):
        endpoint = GAME_ENDPOINTS.get(game_key, "WinGo_30S")
        url = f"{GAME_API_BASE}/{endpoint}/GetHistoryIssuePage.json?pageSize=50&pageNo=1"
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

    def get_all(self, game_key="30s"):
        with self.lock:
            data = self.cache.get(game_key, [])
            return [r["number"] for r in data], [r["issue"] for r in data]

    def init_cache(self):
        for game in GAME_ENDPOINTS:
            self.update(game)
            time.sleep(0.2)

buffer = GameDataBuffer()

# ============================================================
# MODEL 1: MARKOV CHAIN
# ============================================================

class MarkovModel:
    def __init__(self, order=2):
        self.order = order
        self.transitions = {}
        self.transitions_1 = {}
        self.unigram = Counter()

    def train(self, numbers):
        self.transitions = {}
        for i in range(len(numbers) - self.order):
            state = tuple(numbers[i:i + self.order])
            next_num = numbers[i + self.order]
            if state not in self.transitions:
                self.transitions[state] = Counter()
            self.transitions[state][next_num] += 1
        # Also train order-1 as fallback
        self.transitions_1 = {}
        for i in range(len(numbers) - 1):
            state = (numbers[i],)
            next_num = numbers[i + 1]
            if state not in self.transitions_1:
                self.transitions_1[state] = Counter()
            self.transitions_1[state][next_num] += 1
        # Unigram fallback
        self.unigram = Counter(numbers[-20:])

    def predict(self, numbers):
        # Try order-2 first
        if len(numbers) >= self.order:
            state = tuple(numbers[-self.order:])
            if state in self.transitions:
                total = sum(self.transitions[state].values())
                probs = {}
                for n in range(10):
                    probs[n] = round(self.transitions[state].get(n, 0) / total * 100, 1)
                best = max(self.transitions[state], key=self.transitions[state].get)
                return best, probs
        # Fallback order-1
        if len(numbers) >= 1:
            state = (numbers[-1],)
            if state in self.transitions_1:
                total = sum(self.transitions_1[state].values())
                probs = {}
                for n in range(10):
                    probs[n] = round(self.transitions_1[state].get(n, 0) / total * 100, 1)
                best = max(self.transitions_1[state], key=self.transitions_1[state].get)
                return best, probs
        # Fallback unigram
        if self.unigram:
            total = sum(self.unigram.values())
            probs = {}
            for n in range(10):
                probs[n] = round(self.unigram.get(n, 0) / total * 100, 1)
            best = self.unigram.most_common(1)[0][0]
            return best, probs
        return None, {}

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_features(numbers, idx, window=10):
    if idx < window:
        return None
    w = numbers[idx - window:idx + 1]
    feats = []

    # Last number
    feats.append(w[-1])

    # Frequency of each digit in window (10 features)
    for d in range(10):
        feats.append(w.count(d))

    # Position of last occurrence of each digit (10 features)
    for d in range(10):
        pos = -1
        for i in range(len(w) - 1, -1, -1):
            if w[i] == d:
                pos = len(w) - 1 - i
                break
        feats.append(pos if pos >= 0 else window + 1)

    # Gap since each digit last appeared (10 features)
    for d in range(10):
        gap = 0
        for i in range(len(w) - 1, -1, -1):
            if w[i] == d:
                break
            gap += 1
        feats.append(gap)

    # Sum features
    feats.append(sum(w[-2:]))
    feats.append(sum(w[-3:]))
    feats.append(sum(w[-5:]))

    # Mean
    feats.append(round(np.mean(w), 2))

    # Std
    feats.append(round(np.std(w), 2))

    # Is even/odd
    feats.append(w[-1] % 2)
    feats.append(sum(w[-2:]) % 2)
    feats.append(sum(w[-3:]) % 2)

    # Repeat count
    streak = 0
    for i in range(len(w) - 1, -1, -1):
        if w[i] == w[-1]:
            streak += 1
        else:
            break
    feats.append(streak)

    # Diff features
    feats.append(abs(w[-1] - w[-2]))
    feats.append(abs(w[-1] - w[-3]))

    return feats

# ============================================================
# MODEL 2: XGBOOST
# ============================================================

class XGBoostModel:
    def __init__(self):
        self.model = None
        self.trained = False

    def train(self, numbers, window=10):
        X, y = [], []
        min_samples = min(30, len(numbers) - window - 1)
        for i in range(max(window, 5), len(numbers) - 1):
            feat = build_features(numbers, i, window)
            if feat is not None:
                X.append(feat)
                y.append(numbers[i + 1])
        if len(X) < 10:
            return False
        X = np.array(X)
        y = np.array(y)
        self.model = XGBClassifier(
            n_estimators=min(200, max(50, len(X))),
            max_depth=min(6, max(3, len(X) // 10)),
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            verbosity=0
        )
        self.model.fit(X, y)
        self.trained = True
        return True

    def predict(self, numbers, window=10):
        if not self.trained or len(numbers) < window:
            return None, {}
        feat = build_features(numbers, len(numbers) - 1, window)
        if feat is None:
            return None, {}
        X = np.array([feat])
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        prob_dict = {}
        for i, c in enumerate(classes):
            prob_dict[int(c)] = round(float(probs[i]) * 100, 1)
        # Fill missing
        for n in range(10):
            if n not in prob_dict:
                prob_dict[n] = 0.0
        best = int(classes[np.argmax(probs)])
        return best, prob_dict

# ============================================================
# MODEL 3: RANDOM FOREST + GRADIENT BOOSTING (ENSEMBLE)
# ============================================================

class SklearnEnsemble:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42)
        self.trained = False

    def train(self, numbers, window=10):
        X, y = [], []
        for i in range(max(window, 5), len(numbers) - 1):
            feat = build_features(numbers, i, window)
            if feat is not None:
                X.append(feat)
                y.append(numbers[i + 1])
        if len(X) < 10:
            return False
        X = np.array(X)
        y = np.array(y)
        self.rf = RandomForestClassifier(n_estimators=min(200, max(50, len(X))), max_depth=min(8, max(3, len(X) // 10)), random_state=42)
        self.gb = GradientBoostingClassifier(n_estimators=min(150, max(30, len(X))), max_depth=min(5, max(2, len(X) // 15)), random_state=42)
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        self.trained = True
        return True

    def predict(self, numbers, window=10):
        if not self.trained or len(numbers) < window:
            return None, {}
        feat = build_features(numbers, len(numbers) - 1, window)
        if feat is None:
            return None, {}
        X = np.array([feat])

        # RF prediction
        rf_probs = self.rf.predict_proba(X)[0]
        rf_classes = self.rf.classes_

        # GB prediction
        gb_probs = self.gb.predict_proba(X)[0]
        gb_classes = self.gb.classes_

        # Average probabilities
        avg_probs = {}
        for n in range(10):
            p1 = 0.0
            p2 = 0.0
            if n in rf_classes:
                p1 = float(rf_probs[list(rf_classes).index(n)])
            if n in gb_classes:
                p2 = float(gb_probs[list(gb_classes).index(n)])
            avg_probs[n] = round((p1 + p2) / 2 * 100, 1)

        best = max(avg_probs, key=avg_probs.get)
        return best, avg_probs

# ============================================================
# MAJORITY VOTING ENSEMBLE
# ============================================================

class EnsemblePredictor:
    def __init__(self):
        self.markov = MarkovModel(order=2)
        self.xgb = XGBoostModel()
        self.sklearn = SklearnEnsemble()
        self.ready = False
        self.pred_count = 0
        self.retrain_interval = 5

    def train_all(self, numbers):
        print("  [TRAINING] Markov...")
        self.markov.train(numbers)
        print("  [TRAINING] XGBoost...")
        ok1 = self.xgb.train(numbers)
        print(f"  [XGBoost] {'OK' if ok1 else 'FAILED'}")
        print("  [TRAINING] Sklearn Ensemble (RF + GB)...")
        ok2 = self.sklearn.train(numbers)
        print(f"  [Sklearn] {'OK' if ok2 else 'FAILED'}")
        self.ready = True
        print("  [READY] All models trained!")

    def predict(self, numbers, game="30s"):
        try:
            if not self.ready:
                return {"error": "Models not trained"}
            if len(numbers) < 10:
                return {"error": f"Need at least 10 numbers, got {len(numbers)}"}

            # Get predictions from all 3 models
            m1_num, m1_probs = None, {}
            m2_num, m2_probs = None, {}
            m3_num, m3_probs = None, {}

            try:
                m1_num, m1_probs = self.markov.predict(numbers)
            except Exception as e:
                print(f"  [MARKOV ERROR] {e}")

            try:
                m2_num, m2_probs = self.xgb.predict(numbers)
            except Exception as e:
                print(f"  [XGBOOST ERROR] {e}")

            try:
                m3_num, m3_probs = self.sklearn.predict(numbers)
            except Exception as e:
                print(f"  [SKLEARN ERROR] {e}")

            votes = {}
            all_probs = {}

            if m1_num is not None:
                votes["markov"] = int(m1_num)
                all_probs["markov"] = {int(k): float(v) for k, v in m1_probs.items()}
            if m2_num is not None:
                votes["xgboost"] = int(m2_num)
                all_probs["xgboost"] = {int(k): float(v) for k, v in m2_probs.items()}
            if m3_num is not None:
                votes["sklearn"] = int(m3_num)
                all_probs["sklearn"] = {int(k): float(v) for k, v in m3_probs.items()}

            if not votes:
                return {"error": "All models failed", "prediction": "0", "number": 0,
                        "suggested_numbers": [0, 1, 2, 3, 4], "confidence": 0}

            # --- Majority voting ---
            vote_counts = Counter(votes.values())
            majority_num = int(vote_counts.most_common(1)[0][0])
            majority_count = vote_counts.most_common(1)[0][1]

            # --- Combined probability (average all models) ---
            combined_probs = {}
            for n in range(10):
                p_list = []
                for model_name, probs in all_probs.items():
                    if n in probs:
                        p_list.append(probs[n])
                combined_probs[n] = round(float(np.mean(p_list)), 1) if p_list else 0.0

            # --- Confidence ---
            confidence = round(majority_count / max(len(votes), 1) * 100, 1)

            # --- Top 5 numbers with BIG/SMALL majority ---
            sorted_nums = sorted(combined_probs, key=combined_probs.get, reverse=True)

            big_score = sum(combined_probs.get(n, 0) for n in range(5, 10))
            small_score = sum(combined_probs.get(n, 0) for n in range(0, 5))
            tf_hint = "BIG" if big_score > small_score else "SMALL"

            if tf_hint == "BIG":
                big_nums = [n for n in sorted_nums if n >= 5]
                small_nums = [n for n in sorted_nums if n <= 4]
                top5 = big_nums[:3] + small_nums[:2]
            else:
                small_nums = [n for n in sorted_nums if n <= 4]
                big_nums = [n for n in sorted_nums if n >= 5]
                top5 = small_nums[:3] + big_nums[:2]

            if len(top5) < 5:
                top5 += [n for n in sorted_nums if n not in top5][:5 - len(top5)]

            return {
                "prediction": str(majority_num),
                "number": majority_num,
                "suggested_numbers": top5,
                "confidence": confidence,
                "models": votes,
                "combined_probs": combined_probs,
                "tf_hint": tf_hint,
                "vote_counts": dict(vote_counts),
                "game": game,
                "sequence": numbers[-10:],
            }
        except Exception as e:
            print(f"  [PREDICT ERROR] {e}")
            return {"error": str(e), "prediction": "0", "number": 0,
                    "suggested_numbers": [0, 1, 2, 3, 4], "confidence": 0}

    def auto_predict(self, game="30s"):
        try:
            buffer.update(game)
            nums, issues = buffer.get_numbers(game, count=50)
            nums_display, issues_display = buffer.get_numbers(game, count=10)
            if len(nums) < 10:
                return {"error": "Not enough data", "cached": len(nums),
                        "prediction": "0", "number": 0, "suggested_numbers": [0,1,2,3,4]}
            result = self.predict(nums, game)
            result["source"] = "bright-host-spot_live"
            result["cached_records"] = len(nums)
            result["latest_period"] = issues[-1] if issues else "unknown"
            result["recent_numbers"] = nums_display

            # Retrain every 5 predictions
            self.pred_count += 1
            if self.pred_count >= self.retrain_interval:
                self.pred_count = 0
                try:
                    all_nums, _ = buffer.get_all(game)
                    if len(all_nums) >= 10:
                        self.train_all(all_nums)
                        result["retrained"] = True
                except Exception as e:
                    print(f"  [RETRAIN ERROR] {e}")

            return result
        except Exception as e:
            print(f"  [AUTO PREDICT ERROR] {e}")
            return {"error": str(e), "prediction": "0", "number": 0,
                    "suggested_numbers": [0,1,2,3,4], "confidence": 0}

predictor = EnsemblePredictor()

# ============================================================
# AUTO RETRAIN (every 5 minutes)
# ============================================================

def auto_retrain():
    while True:
        time.sleep(400)
        try:
            for game in ["30s", "1m"]:
                nums, _ = buffer.get_all(game)
                if len(nums) >= 30:
                    predictor.train_all(nums)
                    print(f"  [AUTO RETRAIN] {game} - {len(nums)} records")
        except Exception as e:
            print(f"  [RETRAIN ERROR] {e}")

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "name": "WinGo Predictor v9 - Markov + XGBoost + Sklearn",
        "version": "9.0",
        "status": "ready" if predictor.ready else "loading",
        "models": {
            "markov": True,
            "xgboost": predictor.xgb.trained,
            "sklearn_rf_gb": predictor.sklearn.trained,
        },
        "buffer_size": len(buffer.cache.get("30s", [])),
        "endpoints": {
            "GET /": "This page",
            "GET /status": "Model status",
            "GET /predict/auto?game=30s": "Live predict (3 models + majority vote)",
            "GET /predict?game=30s&numbers=1,2,3,...": "Predict with your numbers",
            "GET /buffer?game=30s": "View cached numbers",
            "GET /refresh?game=30s": "Force refresh",
        }
    })

@app.route('/status')
def status():
    return jsonify({
        "ready": predictor.ready,
        "models": {
            "markov": True,
            "xgboost": predictor.xgb.trained,
            "sklearn_rf_gb": predictor.sklearn.trained,
        },
        "version": "9.0",
        "buffer": {k: len(v) for k, v in buffer.cache.items()},
    })

@app.route('/buffer')
def show_buffer():
    game = request.args.get('game', '30s')
    nums, issues = buffer.get_numbers(game, count=25)
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
    nums, _ = buffer.get_numbers(game, count=25)
    return jsonify({"game": game, "count": len(nums), "latest": nums[0] if nums else "none"})

@app.route('/predict/auto')
def auto_predict():
    try:
        game = request.args.get('game', '30s')
        return jsonify(predictor.auto_predict(game))
    except Exception as e:
        return jsonify({"error": str(e), "prediction": "0", "number": 0,
                        "suggested_numbers": [0,1,2,3,4]})

@app.route('/predict')
def predict_get():
    try:
        game = request.args.get('game', '30s')
        nums_str = request.args.get('numbers', '')
        if not nums_str:
            return jsonify(predictor.auto_predict(game))
        numbers = [int(x.strip()) for x in nums_str.split(',') if x.strip().isdigit()]
        return jsonify(predictor.predict(numbers, game))
    except Exception as e:
        return jsonify({"error": str(e), "prediction": "0", "number": 0,
                        "suggested_numbers": [0,1,2,3,4]})

@app.route('/predict', methods=['POST'])
def predict_post():
    try:
        data = request.get_json() or {}
        numbers = data.get('numbers', [])
        game = data.get('game', '30s')
        if not numbers:
            return jsonify(predictor.auto_predict(game))
        return jsonify(predictor.predict(numbers, game))
    except Exception as e:
        return jsonify({"error": str(e), "prediction": "0", "number": 0,
                        "suggested_numbers": [0,1,2,3,4]})

@app.errorhandler(500)
def handle_500(e):
    return jsonify({"error": "Internal error", "prediction": "0", "number": 0,
                    "suggested_numbers": [0,1,2,3,4]}), 200

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Not found", "prediction": "0", "number": 0,
                    "suggested_numbers": [0,1,2,3,4]}), 200

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
    print("  WinGo Predictor v9 - Markov + XGBoost + Sklearn")
    print("  3 AI Models + Majority Voting")
    print("=" * 50)
    print("  Initializing buffer...")
    buffer.init_cache()

    # Train on all games
    for game in ["30s", "1m"]:
        nums, _ = buffer.get_all(game)
        if len(nums) >= 10:
            print(f"  Training on {game} ({len(nums)} records)...")
            predictor.train_all(nums)
            break
    else:
        # Fallback: train on whatever we have
        nums, _ = buffer.get_all("30s")
        if len(nums) >= 10:
            predictor.train_all(nums)

    t1 = threading.Thread(target=auto_refresh, daemon=True)
    t1.start()
    t2 = threading.Thread(target=auto_retrain, daemon=True)
    t2.start()

    port = int(os.environ.get('PORT', 5000))
    print(f"  Port: {port} | Ready: {predictor.ready}")
    print("=" * 50)
    app.run(host='0.0.0.0', port=port, debug=False)
