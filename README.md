README.md
📘 Course Recommender System
The Course Recommender System is a modular, API‑driven recommendation engine built using content‑based filtering, Bag‑of‑Words similarity, and a simple user‑based collaborative filtering baseline.
It is designed for clarity, extensibility, and production readiness, with a clean separation between:

Backend API (backend_app.py)

Recommender logic (recommender.py)

Utility modules (/src)

Datasets (/data)

This project is based on the IBM ML321 Recommender Systems labs and rebuilt into a standalone backend suitable for deployment or portfolio use.

🚀 Features
Content‑Based Filtering  
Computes similarity between courses using Bag‑of‑Words vectors.

Cosine Similarity Engine  
Fast vector comparison using SciPy.

User‑Based Recommendations  
Simple baseline using global popularity among unrated items.

Modular Architecture  
Backend and recommender logic are fully separated.

API‑Ready Backend  
Flask‑based API with clean, predictable endpoints.

Extensible Utility Modules  
Includes reusable helpers for BoW alignment, similarity, and data loading.

📁 Project Structure
Code
course-recommender/
│
├── backend_app.py          # API backend (Flask)
├── recommender.py          # Core recommendation logic
│
├── data/                   # Required datasets
│   ├── courses.csv
│   ├── ratings.csv
│   └── courses_bows.csv
│
├── src/                    # Utility modules
│   ├── data_loader.py
│   ├── bow_utils.py
│   ├── similarity.py
│   └── model_utils.py
│
├── notebooks/              # Optional: lab reconstruction
│   └── lab_rebuild.ipynb
│
├── figures/                # Optional: plots or diagrams
│
├── README.md
├── requirements.txt
└── .gitignore
📦 Installation
1. Clone the repository
Code
git clone https://github.com/<your-username>/course-recommender.git
cd course-recommender
2. Create a virtual environment
Code
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
3. Install dependencies
Code
pip install -r requirements.txt
▶️ Running the Backend
Start the Flask API:

Code
python backend_app.py
The server will run at:

Code
http://localhost:8000
🔍 API Endpoints
1. Health Check
Code
GET /health
Returns:

json
{"status": "ok"}
2. Similar Courses (Content‑Based)
Code
GET /similar/<course_id>?top_n=10
Example:

Code
/similar/ML0101ENv3
Returns:

course_id

title

description

similarity score

3. User Recommendations (CF Baseline)
Code
GET /recommend/<user_id>?top_n=10
Returns:

course_id

title

description

score

📚 Data Sources
All datasets originate from IBM Skills Network:

course_processed.csv → renamed to courses.csv

courses_bows.csv

ratings.csv

These files must be placed in the /data directory.

🧩 How It Works
Content‑Based Filtering
Uses:

Bag‑of‑Words vectors (courses_bows.csv)

Cosine similarity

Vector alignment via pivot_two_bows()

User‑Based Recommendations
Simple baseline:

Remove courses the user already rated

Rank remaining courses by global popularity

🛠 Extensibility
You can extend the system with:

TF‑IDF embeddings

Sentence‑transformer embeddings

Hybrid recommender (content + CF)

Neural collaborative filtering

Frontend UI (React, Streamlit, etc.)

🧪 Testing
A pytest suite can be added to validate:

cosine similarity correctness

BoW pivot alignment

API endpoint responses

dataset loading

📄 License
MIT License (recommended for open-source projects).

🙌 Acknowledgements
This project is based on the IBM ML321 Recommender Systems course, adapted into a standalone backend architecture.
