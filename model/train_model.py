"""
Sentiment Analysis Model Training
==================================
LSTM-equivalent deep learning model using sklearn MLP with word embeddings.
Simulates TensorFlow/Keras LSTM pipeline with identical preprocessing.

FIXES:
  - Uses SentimentPipeline from sentiment_core.py (same class app.py loads)
  - Paths are relative so they work on Windows and Linux
  - Tokenizer now handles emojis and ! ? signals (via sentiment_core)
"""

import os
import sys
import json
import time
import random
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, accuracy_score
)

warnings.filterwarnings('ignore')

# ── Path fix: always resolve relative to this file ────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))   # model/
ROOT      = os.path.dirname(HERE)                         # project root
SAVE_DIR  = os.path.join(HERE, "saved")
DATA_DIR  = os.path.join(ROOT, "data")

# ── Import from sentiment_core (same classes app.py uses) ────────────────────
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "sentiment_core", os.path.join(ROOT, "sentiment_core.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sentiment_core"] = _mod
_spec.loader.exec_module(_mod)

SentimentPipeline = _mod.SentimentPipeline
LABELS            = _mod.LABELS
VOCAB_SIZE        = _mod.VOCAB_SIZE
MAX_LEN           = _mod.MAX_LEN
EMBED_DIM         = _mod.EMBED_DIM

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# SYNTHETIC DATASET
# Now includes emoji + punctuation examples so
# the model actually learns from them.
# ─────────────────────────────────────────────
POSITIVE_TEMPLATES = [
    "I absolutely love {noun}, it's {adj} and amazing!",
    "This {noun} is fantastic, highly recommend to everyone.",
    "Excellent {noun}! Really exceeded my expectations today.",
    "The {noun} was superb and the experience was wonderful.",
    "Brilliant {noun}! I'm so happy with the results.",
    "Great quality {noun}, very satisfied with my purchase.",
    "Outstanding {noun}! Will definitely buy again soon.",
    "Wonderful {noun} experience, totally worth every penny.",
    "Love this {noun}, it works perfectly as described.",
    "Incredible {noun}! Best decision I've made this year.",
    "Very happy with the {noun}, top notch quality service.",
    "Amazing {noun}! Shipping was fast and product is perfect.",
    "So pleased with the {noun}, exceeded all my expectations.",
    "Delightful {noun}, came quickly and works flawlessly.",
    "This {noun} is exactly what I needed, very impressed.",
    "Superb {noun} quality, the team did a fantastic job.",
    "Thrilled with the {noun}! Customer service was helpful.",
    "The {noun} is beautiful and well made, very impressed.",
    "Couldn't be happier with my {noun}, perfect in every way.",
    "Phenomenal {noun}! 5 stars without any hesitation.",
    # Emoji-rich positive
    "Best {noun} ever!!! 😍 Absolutely love it so much!",
    "This {noun} is amazing 😊 would 100% recommend to friends.",
    "So happy with the {noun} 🎉 arrived fast and works great!",
    "Love love love this {noun}! 👍 Worth every single penny!",
    "The {noun} is perfect 💯 exactly what I was looking for!",
]

NEGATIVE_TEMPLATES = [
    "Terrible {noun}, completely disappointed with the quality.",
    "This {noun} is awful, broke after just one day of use.",
    "Worst {noun} ever, total waste of money I regret.",
    "Horrible experience with the {noun}, never buying again.",
    "Disgusting {noun} quality, feels very cheap and flimsy.",
    "Very disappointed with this {noun}, does not work at all.",
    "Pathetic {noun}! The {adj} quality is absolutely unacceptable.",
    "Dreadful {noun} experience, customer service was rude.",
    "Regret buying this {noun}, it stopped working immediately.",
    "Poor quality {noun}, not as described and very misleading.",
    "Broken {noun} arrived, packaging was terrible and damaged.",
    "This {noun} is a scam, don't waste your hard earned money.",
    "Utterly useless {noun}, returned it for a full refund.",
    "Defective {noun} from day one, very poor craftsmanship.",
    "Never again! This {noun} is the worst purchase ever made.",
    "Extremely disappointed with this {noun}, quality is appalling.",
    "Garbage {noun}! Stopped functioning after just one week.",
    "Lousy {noun}, does not match the description at all.",
    "The {noun} is a disaster, falling apart at the seams.",
    "Terrible {noun} support, they refused to help me whatsoever.",
    # Emoji-rich negative (the exact pattern user reported as broken)
    "Worst {noun} ever!!! 😡 Complete waste of money!!!",
    "This {noun} is garbage 😠 broke on the very first day!!!",
    "Absolutely terrible {noun}!!! 🤬 Never buying this again!!!",
    "Disgusting {noun} 👎 worst purchase I have ever made!!!",
    "Horrible {noun}!!! 😞 So disappointed, does not work at all!!!",
    "The {noun} broke immediately 😤 total scam do not buy!!!",
    "Worst experience ever with {noun}!!! 💔 Avoid at all costs!!!",
    "Pathetic {noun} 😣 stopped working after just one day!!!",
]

NEUTRAL_TEMPLATES = [
    "The {noun} is okay, nothing special but does its job.",
    "Average {noun}, meets basic expectations nothing more.",
    "The {noun} works as described, pretty standard product.",
    "Decent {noun} for the price, neither great nor terrible.",
    "The {noun} is fine, got what I paid for I suppose.",
    "Okay {noun}, arrived on time and in adequate condition.",
    "The {noun} functions correctly, average build quality.",
    "Fair {noun} for the cost, does what it says it will.",
    "Normal {noun} experience, nothing to complain or praise.",
    "Acceptable {noun} overall, could be better could be worse.",
    "The {noun} is standard quality, meets minimum requirements.",
    "Received the {noun} in good condition, seems average quality.",
    "The {noun} does its job, nothing extraordinary to report.",
    "Mediocre {noun} at best, but functional for basic needs.",
    "The {noun} is passable, works fine for occasional use.",
    "Regular {noun} product, similar to others in same range.",
    "The {noun} arrived as expected, standard packaging used.",
    "Typical {noun}, fulfills purpose without standing out.",
    "The {noun} is serviceable, adequate for the price point.",
    "Moderate quality {noun}, acceptable for everyday simple use.",
    # Emoji-neutral
    "The {noun} is okay 😐 nothing special to say about it.",
    "Average {noun}, does the job I guess 🙂 nothing more.",
    "The {noun} arrived fine, pretty standard 🤔 not impressed.",
]

NOUNS = ["product", "service", "experience", "item", "purchase", "delivery",
         "quality", "package", "order", "device", "app", "software", "tool",
         "system", "support", "team", "result", "performance", "feature"]

POSITIVE_ADJ = ["excellent", "wonderful", "fantastic", "brilliant", "superb"]
NEGATIVE_ADJ = ["terrible", "awful", "horrible", "dreadful", "appalling"]
NEUTRAL_ADJ  = ["average", "standard", "typical", "ordinary", "basic"]

POSITIVE_TEMPLATES = [
    # ... existing templates ...

    # Short phrases (critical for real-world input)
    "I love this {noun}!",
    "Love this so much!",
    "I love this 😍",
    "Love it! 😊",
    "I love this {noun} 😍 so good!",
    "This is great 😊 really happy with it!",
    "Love love love this {noun}! 👍",
    "I love this, best {noun} ever!",
    "Loving this {noun} so much right now!",
    "I love this {noun}, makes me so happy!",
]

# Short negative phrases
NEGATIVE_TEMPLATES= [
    "I hate this {noun}!",
    "Hate it! 😠",
    "This is terrible 😡",
    "I hate this so much!!!",
    "Worst {noun} ever! 😤",
]

POSITIVE_TEMPLATES = [
    # Real-world natural sentences
    "Amazon delivery was fast and packaging was great.",
    "Delivery was quick and the packaging was excellent.",
    "Fast shipping and the product was well packaged.",
    "The product arrived quickly and was packaged perfectly.",
    "Shipping was super fast and everything arrived intact.",
    "Great delivery speed and the packaging was very good.",
    "The order came on time and packaging was fantastic.",
    "Quick delivery and well packed, very ha`ppy overall.",
    "Arrived ahead of schedule and packaging was superb.",
    "Fast delivery, great packaging, very satisfied customer.",
]

NEUTRAL_TEMPLATES =[
    # Mixed / complex sentiment
    "The {noun} started slow but ended up being great.",
    "Didn't like it at first but the {noun} grew on me.",
    "The {noun} had issues initially but improved a lot.",
    "Started off bad but the {noun} turned out okay.",
    "The beginning was rough but the {noun} ended well.",
    "Not great at first but the {noun} was good overall.",
]

def generate_dataset(n_per_class=2000):
    texts, labels = [], []
    templates_labels = [
        (NEGATIVE_TEMPLATES, NEGATIVE_ADJ, 0),
        (NEUTRAL_TEMPLATES,  NEUTRAL_ADJ,  1),
        (POSITIVE_TEMPLATES, POSITIVE_ADJ, 2),
    ]
    for templates, adjs, label in templates_labels:
        for _ in range(n_per_class):
            t    = random.choice(templates)
            noun = random.choice(NOUNS)
            adj  = random.choice(adjs)
            text = t.format(noun=noun, adj=adj)
            if random.random() < 0.1:
                text = text.lower()
            if random.random() < 0.05:
                text = text + " " + random.choice(["!", "...", "??"])
            texts.append(text)
            labels.append(label)
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    return list(texts), list(labels)


# ─────────────────────────────────────────────
# EVALUATION & VISUALISATION
# ─────────────────────────────────────────────
def evaluate_model(model, X_test_texts, y_test, save_dir):
    print("\n── Evaluation ──────────────────────────────")
    y_pred = model.predict(X_test_texts)

    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {p*100:.2f}%")
    print(f"  Recall    : {r*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABELS))

    metrics = {
        'accuracy':  round(acc, 4),
        'precision': round(p,   4),
        'recall':    round(r,   4),
        'f1_score':  round(f1,  4),
        'confusion_matrix': cm.tolist(),
        'per_class': {}
    }
    p_cls, r_cls, f1_cls, sup = precision_recall_fscore_support(y_test, y_pred)
    for i, label in enumerate(LABELS):
        metrics['per_class'][label] = {
            'precision': round(float(p_cls[i]),  4),
            'recall':    round(float(r_cls[i]),  4),
            'f1_score':  round(float(f1_cls[i]), 4),
            'support':   int(sup[i]),
        }

    # ── Confusion Matrix Plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=LABELS, yticklabels=LABELS,
        ax=axes[0], linewidths=0.5,
        annot_kws={'size': 14, 'weight': 'bold'},
    )
    axes[0].set_title('Confusion Matrix', color='white', fontsize=14, pad=12)
    axes[0].set_xlabel('Predicted', color='#aaa')
    axes[0].set_ylabel('Actual',    color='#aaa')
    axes[0].tick_params(colors='white')
    axes[0].set_facecolor('#1a1a2e')
    for spine in axes[0].spines.values():
        spine.set_edgecolor('#333')

    # ── Per-Class Metrics Bar ──
    bar_metrics = ['Precision', 'Recall', 'F1-Score']
    bar_values  = [
        [metrics['per_class'][l]['precision'] for l in LABELS],
        [metrics['per_class'][l]['recall']    for l in LABELS],
        [metrics['per_class'][l]['f1_score']  for l in LABELS],
    ]
    x    = np.arange(len(LABELS))
    w    = 0.25
    clrs = ['#e74c3c', '#f39c12', '#2ecc71']
    for i, (metric, vals, c) in enumerate(zip(bar_metrics, bar_values, clrs)):
        axes[1].bar(x + i*w, vals, w, label=metric, color=c, alpha=0.85)
    axes[1].set_title('Per-Class Metrics', color='white', fontsize=14, pad=12)
    axes[1].set_xticks(x + w)
    axes[1].set_xticklabels(LABELS, color='white')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(facecolor='#1a1a2e', labelcolor='white')
    axes[1].set_facecolor('#1a1a2e')
    axes[1].tick_params(colors='white')
    for spine in axes[1].spines.values():
        spine.set_edgecolor('#333')

    plt.tight_layout()
    chart_path = os.path.join(save_dir, "evaluation_charts.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    plt.close()
    print(f"  Charts saved → {chart_path}")

    return metrics


# ─────────────────────────────────────────────
# TRAINING ENTRY POINT
# ─────────────────────────────────────────────
def train_and_save():
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print("═" * 55)
    print("  SENTIMENT ANALYSIS — LSTM MODEL TRAINING")
    print("═" * 55)

    # Generate data
    print("\n[STEP 1] Generating dataset...")
    texts, labels = generate_dataset(n_per_class=2000)
    n     = len(texts)
    split = int(n * 0.8)
    X_train, X_test = texts[:split], texts[split:]
    y_train, y_test = labels[:split], labels[split:]
    print(f"  Total: {n} | Train: {len(X_train)} | Test: {len(X_test)}")

    # Build pipeline using sentiment_core (same class app.py loads)
    print("\n[STEP 2] Fitting tokenizer...")
    pipeline = SentimentPipeline()
    pipeline.tokenizer.fit_on_texts(X_train)
    print(f"  Vocabulary size: {len(pipeline.tokenizer.word_index)}")

    # Train
    print("\n[STEP 3] Training model...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  Training time: {time.time()-t0:.1f}s")

    # Evaluate
    print("\n[STEP 4] Evaluating on test set...")
    metrics = evaluate_model(pipeline, X_test, y_test, save_dir=DATA_DIR)

    # Save
    print("\n[STEP 5] Saving model artifacts...")
    prefix = os.path.join(SAVE_DIR, "sentiment_model")
    pipeline.save(prefix)

    # Save metrics
    metrics_path = os.path.join(DATA_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    print("\n" + "═" * 55)
    print(f"  ✓ Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  ✓ F1-Score: {metrics['f1_score']*100:.2f}%")
    print(f"  ✓ Artifacts saved to: {SAVE_DIR}")
    print("═" * 55)

    return pipeline, metrics


if __name__ == "__main__":
    train_and_save()