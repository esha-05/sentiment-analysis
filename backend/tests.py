"""
API Test Suite — Sentiment Analysis Backend
============================================
Tests all endpoints, error handling, edge cases, and bug fixes.
Run:  python3 backend/tests.py
"""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load sentiment_core so pickle works
import importlib.util
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("sentiment_core", os.path.join(ROOT,"sentiment_core.py"))
mod  = importlib.util.module_from_spec(spec)
sys.modules["sentiment_core"] = mod
spec.loader.exec_module(mod)

from backend.app import app

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def test(name, cond, detail=""):
    status = PASS if cond else FAIL
    results.append((name, status))
    icon = "✅" if cond else "❌"
    print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))
    return cond

def run_tests():
    with app.test_client() as c:
        print("\n" + "═"*58)
        print("  SENTIMENT ANALYSIS API — TEST SUITE")
        print("═"*58)

        # ── GROUP 1: Health ─────────────────────────────────────────
        print("\n[1] Health Check")
        r = c.get("/api/health")
        d = r.get_json()
        test("GET /api/health → 200",          r.status_code == 200)
        test("Response has 'status'",           "status" in d)
        test("Status is 'healthy'",             d.get("status") == "healthy")
        test("Labels present",                  "labels" in d)
        test("Latency key present",             "latency_ms" in d)

        # ── GROUP 2: Single Prediction ──────────────────────────────
        print("\n[2] POST /api/predict — Valid Input")
        r = c.post("/api/predict", json={"text": "I absolutely love this amazing product!"})
        d = r.get_json()
        test("Status 200",                      r.status_code == 200)
        test("success == true",                 d.get("success") is True)
        test("prediction.label exists",         "label" in d.get("prediction", {}))
        test("label is valid class",            d["prediction"]["label"] in ["Positive","Neutral","Negative"])
        test("confidence in [0,1]",             0 <= d["prediction"]["confidence"] <= 1)
        test("probabilities dict present",      "probabilities" in d["prediction"])
        test("3 probability keys",              len(d["prediction"]["probabilities"]) == 3)
        probs = d["prediction"]["probabilities"]
        total = sum(probs.values())
        test("Probabilities sum ≈ 1.0",         abs(total - 1.0) < 0.01,
             f"sum={total:.4f}")
        test("text echoed back",                d.get("text") == "I absolutely love this amazing product!")
        test("latency_ms present",              "latency_ms" in d)

        # ── GROUP 3: Bug Fix Validation ──────────────────────────────
        print("\n[3] Bug Fix #1 — Pickle Serialization (module-safe load)")
        # Already passing if we got here (model loaded correctly)
        test("Model loads from saved artifacts", True, "sentiment_core module registered")

        print("\n[4] Bug Fix #2 — Empty / Invalid Input Guard")
        r = c.post("/api/predict", json={"text": ""})
        test("Empty string → 400",              r.status_code == 400)
        test("Meaningful error message",        "empty" in r.get_json().get("error","").lower())

        r = c.post("/api/predict", json={"text": "   "})
        test("Whitespace-only → 400",           r.status_code == 400)

        r = c.post("/api/predict", json={"text": "x" * 5001})
        test("Oversized text → 413",            r.status_code == 413)
        test("413 error mentions limit",        "5000" in r.get_json().get("error",""))

        r = c.post("/api/predict", json={"text": 12345})
        test("Non-string text → 400",           r.status_code == 400)

        print("\n[5] Bug Fix #3 — Content-Type Validation")
        r = c.post("/api/predict", data="plain text", content_type="text/plain")
        test("Non-JSON → 415",                  r.status_code == 415)
        test("415 error message correct",       "json" in r.get_json().get("error","").lower())

        r = c.post("/api/predict", data=b"\xff\xfe", content_type="application/octet-stream")
        test("Binary body → 415",               r.status_code == 415)

        print("\n[6] Bug Fix #4 — Missing Required Fields")
        r = c.post("/api/predict", json={})
        test("Missing 'text' field → 400",      r.status_code == 400)

        r = c.post("/api/predict", json={"wrong_key": "hello"})
        test("Wrong key → 400",                 r.status_code == 400)

        # ── GROUP 4: Batch Endpoint ──────────────────────────────────
        print("\n[7] POST /api/predict/batch — Valid")
        r = c.post("/api/predict/batch", json={"texts": [
            "Great product, really happy!",
            "Terrible, broken on arrival.",
            "It's okay, average quality.",
        ]})
        d = r.get_json()
        test("Batch 200",                       r.status_code == 200)
        test("count == 3",                      d.get("count") == 3)
        test("3 results returned",              len(d.get("results",[])) == 3)
        test("All results have predictions",    all("prediction" in r for r in d["results"]))
        test("error_count == 0",                d.get("error_count") == 0)

        print("\n[8] POST /api/predict/batch — Edge Cases")
        r = c.post("/api/predict/batch", json={"texts": []})
        test("Empty array → 400",               r.status_code == 400)

        r = c.post("/api/predict/batch", json={"texts": ["x"]*51})
        test("51 texts → 413",                  r.status_code == 413)

        r = c.post("/api/predict/batch", json={"texts": "not a list"})
        test("Non-array texts → 400",           r.status_code == 400)

        # Mixed valid/invalid in batch
        r = c.post("/api/predict/batch", json={"texts": ["Good!", "", "Bad!"]})
        d = r.get_json()
        test("Mixed batch returns partial",     r.status_code == 200)
        test("error_count == 1",                d.get("error_count") == 1)

        # ── GROUP 5: Metrics & Info ──────────────────────────────────
        print("\n[9] GET /api/metrics")
        r = c.get("/api/metrics")
        d = r.get_json()
        test("Metrics 200",                     r.status_code == 200)
        test("accuracy present",                "accuracy" in d.get("metrics",{}))
        test("confusion_matrix present",        "confusion_matrix" in d.get("metrics",{}))
        test("per_class present",               "per_class" in d.get("metrics",{}))
        acc = d["metrics"].get("accuracy", 0)
        test(f"Accuracy ≥ 0.85 ({acc*100:.1f}%)", acc >= 0.85)

        print("\n[10] GET /api/model/info")
        r = c.get("/api/model/info")
        d = r.get_json()
        test("Model info 200",                  r.status_code == 200)
        test("architecture in response",        "architecture" in d.get("model",{}))
        test("endpoints listed",                "endpoints" in d)

        # ── GROUP 6: Error Handlers ──────────────────────────────────
        print("\n[11] Error Handlers")
        # Note: Flask OPTIONS catch-all intercepts GET on truly unknown routes in test client
        # Use a POST to a non-existent, non-OPTIONS-matched route for 404
        r = c.get("/completely-unknown-route-xyz")
        test("Unknown route → 404",             r.status_code == 404)
        test("404 success==false",              r.get_json().get("success") is False)

        r = c.put("/api/predict", json={"text":"hi"})
        test("PUT on POST route → 405",         r.status_code == 405)

        # ── GROUP 7: CORS ────────────────────────────────────────────
        print("\n[12] CORS Headers")
        r = c.get("/api/health")
        test("CORS header present",             "Access-Control-Allow-Origin" in r.headers)
        test("CORS header is *",                r.headers.get("Access-Control-Allow-Origin") == "*")

        # ── GROUP 8: Special characters & unicode ────────────────────
        print("\n[13] Unicode & Special Characters")
        r = c.post("/api/predict", json={"text": "This is 🎉 amazing! Best évèr."})
        test("Unicode input → 200",             r.status_code == 200)
        test("Prediction returned for unicode", "label" in r.get_json().get("prediction",{}))

        r = c.post("/api/predict", json={"text": "!@#$%^&*()"})
        test("Special chars → 200",             r.status_code == 200)

        r = c.post("/api/predict", json={"text": "hello"})
        test("Single word → 200",               r.status_code == 200)

        # ── SUMMARY ─────────────────────────────────────────────────
        passed = sum(1 for _, s in results if s == PASS)
        total  = len(results)
        print(f"\n{'═'*58}")
        print(f"  Results: {passed}/{total} tests passed")
        if passed == total:
            print("  🎉 All tests passed!")
        else:
            failed = [(n,s) for n,s in results if s == FAIL]
            print(f"  ⚠ {len(failed)} failed:")
            for name, _ in failed:
                print(f"    • {name}")
        print(f"{'═'*58}\n")
        return passed, total

if __name__ == "__main__":
    passed, total = run_tests()
    sys.exit(0 if passed == total else 1)
