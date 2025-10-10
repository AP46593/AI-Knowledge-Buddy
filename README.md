# AI Doc Assist
POC for Spark Hackathon

Run commands

Create virtual env:
python -m venv .venv

activate the env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate

install requirements:
pip install --upgrade pip
pip install streamlit ollama PyMuPDF python-docx openpyxl numpy pandas scikit-learn scipy
pip install sentence-transformers
pip install -r requirements.txt

make sure ollama is running and manifest downloaded:
ollama list
ollama serv 

run streamlit app:
streamlit run streamlit_app.py

