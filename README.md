Wearable-Based Stress Detection Using Physiological Signals
Python · scikit-learn · Streamlit · Groq API · Docker · WESAD Dataset

Rebuilt stress classification system on WESAD dataset using proper Leave-One-Subject-Out (LOSO) cross-validation across 15 subjects, eliminating data leakage from the baseline implementation.
Engineered 46 physiological features from chest signals (ECG, EDA, Temp, Resp, EMG, ACC) including HRV metrics and statistical descriptors using windowed extraction at 700 Hz.
Trained and compared 4 ML models (SVM, Decision Tree, Random Forest, ANN); deployed Random Forest as final model based on LOSO F1 score.
Defined a custom stress index (0–100) combining model confidence probability with EDA physiological weighting, mapped to Low / Moderate / High categories.
Integrated Llama3 (GROQ AI) API to generate personalized stress management recommendations based on predicted stress index — hands-on LLM integration and prompt engineering.
Built and deployed an interactive Streamlit web app with real-time gauge visualization, stress timeline, and AI advice panel; containerized using Docker for consistent cross-system deployment.

Future Scope line:
Planned extensions include smartwatch integration (Apple Watch / Fitbit API) for real-time signal streaming and a manual input mode for heart rate and sleep hours-based prediction.
