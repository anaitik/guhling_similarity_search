3D Mesh Similarity Search

How to Run


Install dependencies:


bash   pip install -r requirements.txt


Put your .stl files inside the data/ folder (create the folder if it doesn't exist).
Start the app:


bash   streamlit run app.py

How to Use


Pick an embedding backend from the sidebar.
Set the indexing parameters (bins, points, weights, etc.) — if unsure what values to use, ask ChatGPT for the best ratio based on your dataset.
Click Build / Rebuild selected index to create the index.
Choose a query (a file from the dataset, an upload, or a text description) and click Search.
