# DISEASE PREDICTION FROM SYMPTOM DESCRIPTONS
This project builds a **smart disease prediction system** that identifies possible diseases based on **symptoms entered by users in natural language**.  
By combining **Natural Language Processing (NLP)** and **semantic similarity techniques**, the model understands free-text descriptions and predicts the most likely diseases with high accuracy.

---

## Overview
Traditional symptom-based diagnosis systems rely on fixed keyword inputs.  
This project improves upon that by enabling **free-text and even voice-based inputs**, allowing users to describe their symptoms naturally — just like they would to a doctor.
The system then uses **sentence embeddings** and **similarity scoring** to match user inputs to diseases in the dataset.

---

## 📁Dataset
- **Total Diseases:** ~80  
- **Samples per Disease:** 40–50 unique text descriptions  
- **Data Type:** Natural language symptom descriptions  
- **Purpose:** Train and evaluate semantic similarity-based disease prediction  

---

## ⚙️ Technologies & Libraries Used
1. **Frontend/UI** : Tkinter (Python GUI), ScrolledText, MessageBox
2. **Data Handling** : pandas 
3. **NLP Processing** : spaCy, SentenceTransformers 
4. **Similarity Matching** : cosine similarity (`sentence-transformers.util`), fuzzy matching (`rapidfuzz`) 
5. **Speech Input** : SpeechRecognition (`speech_recognition` library) 
6. **Utilities** : threading, time, collections.Counter 

---

## Working Process
1. **Input Collection**  
   - User enters or speaks their symptoms via GUI.  
   - The system captures and processes the input text.

2. **Text Preprocessing & Embedding**  
   - Input is cleaned and encoded into sentence embeddings using `SentenceTransformer`.  
   - Similar embeddings are generated for all dataset entries.

3. **Similarity Calculation**  
   - Computes **cosine similarity** between user input and each disease description.  
   - Also applies **fuzzy string matching** (`rapidfuzz`) for robustness.

4. **Prediction & Output**  
   - Aggregates similarity scores and identifies the **top probable diseases**.  
   - Displays them on the Tkinter GUI with confidence levels.

---

##  Features
- 🗣️ **Voice Input Support** using the `speech_recognition` library  
- 🧾 **Free-text Input** (no fixed symptom list required)  
- ⚙️ **Semantic Matching** using transformer embeddings  
- 🧮 **Multi-model Similarity Scoring** (cosine + fuzzy)  
- 🪟 **Interactive Tkinter GUI**

---

## Example Use Case
**User Input:**  
> “I have a sore throat, mild fever, and body ache.”
> 
**Predicted Output:**  
> Possible Diseases:  
> - Common Cold  
> - Influenza  
> - Viral Infection

---

## Future Improvements
- Train on audio based symptoms for **enhanced voice input support**
- Integrate a **medical advice or prevention tips** section  
- Deploy as a **web app (Streamlit/Flask)** for wider accessibility  
- Include **real medical dataset integration**

---

## Author
**Arukshita Dubey**   
📧 [https://www.linkedin.com/in/arukshita-dubey-811b22249/]

---

## 🧾 Citation
If using the idea or data, please cite:
> Arukshita Dubey (2025). *Disease Prediction from Symptom Descriptions* — A Natural Language Processing-based Smart Diagnosis System.
