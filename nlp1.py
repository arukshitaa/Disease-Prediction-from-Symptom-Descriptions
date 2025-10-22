import tkinter as tk
from tkinter import messagebox, scrolledtext
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer, util
from collections import Counter
from rapidfuzz import fuzz
import speech_recognition as sr
import threading
import time

# Load models and data
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("dataset.csv", encoding='ISO-8859-1')
df.dropna(inplace=True)

# Build disease-symptom map
def extract_symptoms(text):
    doc = nlp(text)
    symptoms = set()
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if len(phrase) < 3 or all(token.is_stop for token in chunk):
            continue
        meaningful = [token for token in chunk if not token.is_stop and token.pos_ in {"ADJ", "NOUN"}]
        if not meaningful:
            continue
        if len(meaningful) == 1 and meaningful[0].pos_ == "NOUN":
            continue
        symptoms.add(phrase)
    return list(symptoms)

def build_disease_symptom_map():
    disease_symptom_map = {}
    for label in df['label'].unique():
        texts = df[df['label'] == label]['text'].tolist()
        all_symptoms = []
        for t in texts:
            all_symptoms.extend(extract_symptoms(t))
        counter = Counter(all_symptoms)
        most_common = [sym for sym, _ in counter.most_common(10)]
        disease_symptom_map[label] = most_common
    return disease_symptom_map

def get_top_diseases(user_input, disease_symptom_map, weight_cosine=0.6, weight_fuzzy=0.4):
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    user_symptoms = extract_symptoms(user_input)
    scores = []
    for disease, symptoms in disease_symptom_map.items():
        if not symptoms:
            continue
        joined = ", ".join(symptoms)
        disease_embedding = model.encode(joined, convert_to_tensor=True)
        cosine_score = util.pytorch_cos_sim(user_embedding, disease_embedding).item()

        fuzzy_sum = 0
        for user_symptom in user_symptoms:
            fuzzy_matches = [fuzz.partial_ratio(user_symptom, s) for s in symptoms]
            if fuzzy_matches:
                fuzzy_sum += max(fuzzy_matches)

        fuzzy_score = fuzzy_sum / (len(user_symptoms) * 100) if user_symptoms else 0
        final_score = weight_cosine * cosine_score + weight_fuzzy * fuzzy_score
        scores.append((disease, final_score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:5]

def refined_diagnosis(user_input, follow_up_answers, disease_symptom_map):
    confirmed = [s for s, val in follow_up_answers.items() if val]
    refined_input = user_input + ". " + ", ".join(confirmed)
    top = get_top_diseases(refined_input, disease_symptom_map)
    return top[0]

def get_speech_input():
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True  # Allow auto-adjust based on input
    recognizer.energy_threshold = 250  # Lower threshold to improve distance sensitivity

    with sr.Microphone() as source:
        try:
            print("[INFO] Calibrating mic for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=4)
            print(f"[DEBUG] Energy Threshold after calibration: {recognizer.energy_threshold}")

            print("[INFO] Listening now! Speak clearly from wherever you are.")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=45)
            print("[INFO] Processing your audio...")

            return recognizer.recognize_google(audio)

        except sr.WaitTimeoutError:
            print("[ERROR] Timed out waiting for speech.")
            return ""
        except sr.UnknownValueError:
            print("[ERROR] Could not understand audio.")
            return ""
        except sr.RequestError as e:
            print(f"[ERROR] Recognition service error: {e}")
            return ""

# GUI setup
disease_symptom_map = build_disease_symptom_map()

root = tk.Tk()
root.title("🩺 Disease Diagnosis Assistant")
root.geometry("700x700")

label = tk.Label(root, text="Describe your symptoms (text or voice):", font=("Arial", 14))
label.pack(pady=10)

text_input = scrolledtext.ScrolledText(root, height=5, font=("Arial", 12))
text_input.pack(pady=5, padx=10, fill="x")

result_label = tk.Label(root, text="", font=("Arial", 12), fg="green")
result_label.pack(pady=10)

result_box = scrolledtext.ScrolledText(root, height=10, font=("Arial", 11))
result_box.pack(padx=10, pady=10, fill="both")

followup_frame = tk.Frame(root)
followup_frame.pack(pady=10, fill="x")

final_label = tk.Label(root, text="", font=("Arial", 13, "bold"), fg="blue")
final_label.pack(pady=10)

follow_up_vars = {}

def ask_follow_up_questions(top5, user_input):
    for widget in followup_frame.winfo_children():
        widget.destroy()

    asked = set()
    follow_up_symptoms = []
    for d, _ in top5:
        count = 0
        for symptom in disease_symptom_map[d]:
            if symptom not in asked and len(symptom) > 2 and not any(w in symptom for w in ['my', 'your', 'i', 'lot', 'bit', 'thing','past','week']):
                follow_up_symptoms.append((d, symptom))
                asked.add(symptom)
                count += 1
            if count == 1:
                break
    follow_up_symptoms = follow_up_symptoms[:5]

    tk.Label(followup_frame, text="Answer the following for better accuracy:", font=("Arial", 12)).pack(anchor='w')
    for disease, symptom in follow_up_symptoms:
        var = tk.BooleanVar()
        follow_up_vars[symptom] = var
        cb = tk.Checkbutton(followup_frame, text=f" Have you experienced {symptom}?", variable=var, onvalue=True, offvalue=False, font=("Arial", 11))
        cb.pack(anchor='w')

    submit_btn = tk.Button(followup_frame, text="Submit Follow-Up", command=lambda: display_final_diagnosis(user_input), bg="#FF9800", fg="white", font=("Arial", 12))
    submit_btn.pack(pady=5)

def display_final_diagnosis(user_input):
    follow_up_answers = {sym: var.get() for sym, var in follow_up_vars.items()}
    final_disease, confidence = refined_diagnosis(user_input, follow_up_answers, disease_symptom_map)
    final_label.config(text=f"✅ Final Diagnosis: {final_disease} (Confidence: {confidence:.4f})")

def process_input(user_input):
    if not user_input.strip():
        messagebox.showwarning("Missing Input", "Please enter or speak your symptoms.")
        return
    result_label.config(text="🔎 Analyzing symptoms...")
    root.update()
    top5 = get_top_diseases(user_input, disease_symptom_map)
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, "Top 5 Possible Conditions:\n\n")
    for i, (disease, score) in enumerate(top5, 1):
        result_box.insert(tk.END, f"{i}. {disease} (Score: {score:.4f})\n")
    result_label.config(text="💡 Initial diagnosis ready. Please respond to follow-up:")
    ask_follow_up_questions(top5, user_input)

def on_submit():
    user_input = text_input.get("1.0", tk.END)
    process_input(user_input)

def on_speak():
    result_label.config(text="🎙️ Adjusting for background noise...")
    root.update()

    def thread_func():
        result_label.config(text="🎤 Listening now! Speak your symptoms.")
        root.update()

        spoken = get_speech_input()

        if spoken:
            result_label.config(text=f"✔️ Detected: \"{spoken}\"")
        else:
            result_label.config(text="❌ Couldn't detect speech. Try again.")

        text_input.delete("1.0", tk.END)
        text_input.insert(tk.END, spoken)

        if spoken:
            process_input(spoken)

    threading.Thread(target=thread_func).start()

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

type_btn = tk.Button(button_frame, text="Submit Text", command=on_submit, font=("Arial", 12), bg="#4CAF50", fg="white")
type_btn.pack(side="left", padx=10)

speak_btn = tk.Button(button_frame, text="Speak Symptoms", command=on_speak, font=("Arial", 12), bg="#2196F3", fg="white")
speak_btn.pack(side="left", padx=10)

root.mainloop()