# Driven-Intrusion-Detection-System-For-Network-Security-Using-Machine-Learning
This project focuses on building an Intelligent Intrusion Detection System (IDS) that detects malicious network activities using Machine Learning models such as AlexNet, LSTM, and MiniVGG, along with traditional models like SVM and K-Means.
The system is designed to identify known attacks and zero-day attacks with improved accuracy and reduced false positives.

# 📌 Features

✔ Real-time intrusion detection
✔ Supports signature-based and anomaly-based detection
✔ Ensemble ML models for improved accuracy
✔ Self-learning system using adaptive ML techniques
✔ Scalable for cloud, IoT, and enterprise environments
✔ Tkinter GUI with login and prediction interface
✔ Uses majority voting from three deep-learning models

# 🧠 Machine Learning Models Used

•	AlexNet
•	LSTM
•	MiniVGG
•	SVM
•	K-Means
•	Ensemble learning for combined prediction
Each model uses appropriate preprocessing pipelines (MinMaxScaler, Normalization, etc.).


# 📂 Project Structure

├── data.csv
├── balanced1.csv
├── balanced2.csv
├── finaldataset.csv          # Final cleaned & processed dataset
│
├── model_lstm1.json          # LSTM model architecture
├── lstm_weight1.h5           # LSTM trained weights
├── alexmodel.model           # AlexNet model
├── model.model               # MiniVGG model
│
├── vggscale.pkl              # Scaler for MiniVGG
├── norm.pkl                  # Normalization scaler
├── minmaxlstm.pkl            # MinMaxScaler for LSTM
│
├── dsste.py                  # Data preprocessing, cleaning, balancing
├── gui.py                    # Tkinter GUI interface
│
└── README.md                 # Documentation


# ⚙️ How the System Works

1.	Load models and scalers (AlexNet, LSTM, MiniVGG)
2.	User inputs features through the Tkinter GUI
3.	Features are scaled and reshaped for each model
4.	Each model predicts the attack class
5.	Majority voting determines final classification
6.	Result is displayed as:
o	Normal
o	DOS
o	Probe
o	R2L
o	U2R


 # Technologies Used
 
•	Python 3.11
•	TensorFlow / Keras
•	Scikit-learn
•	Tkinter
•	Pandas, NumPy
•	NetworkX (for analyzing network traffic patterns)
•	Matplotlib
•	Pickle


# 🧪 Model Evaluation

Models are evaluated using:
• Accuracy
• Precision
• Recall
• F1-score
• Confusion Matrix


# 🚀 Future Enhancements

• Real-time packet capture integration
• Cloud-based deployment
• Extended dataset support
• Improved ensemble algorithms
• Auto-retraining using live data (self-learning)

# How to Run

1. Create Virtual Environment
   python3 -m venv venv

2. Navigate to the Project Folder
   cd "C:\Users\YourName\path\to\project"

3. Activate Virtual Environment  
   source venv/bin/activate

4. Launch the IDS GUI
   python gui.py



# 👩‍💻 Author

Devika
MCA Student | Cloud, DevOps & AI Enthusiast
Puttur, India


