# AI Doc Assist
POC for Spark Hackathon

Run commands

Create virtual env:
python -m venv .venv

activate the env
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate

install requirements:
download tesseract-ocr for windows

pip install --upgrade pip
pip install streamlit ollama PyMuPDF python-docx openpyxl numpy pandas scikit-learn scipy
pip install sentence-transformers
pip install -r requirements.txt
pip install pytesseract pillow 

make sure ollama is running and manifest downloaded:
ollama list
ollama pull llama3:8B
ollama pull llama3:latest or llama3:8B
ollama pull nomic-embed-text:latest
ollama pull llava:13b  
ollama serv 

run streamlit app:
streamlit run streamlit_app.py

