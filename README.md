# 3D Mesh Similarity Search

## How to Run

1. Install dependencies:
```bash
   pip install -r requirements.txt
```
2. Put your `.stl` files inside the `data/` folder (create the folder if it doesn't exist).
3. Start the app:
```bash
   streamlit run app.py
```

## How to Use

1. Pick an embedding backend from the sidebar.
(best one to use is hybrid methods , each method requires different building of index)
2. Set the indexing parameters (bins, points, weights, etc.) — if unsure what values to use, ask ChatGPT for the best ratio based on your dataset.
if you have build a index ,please when u use it ,put the parameters same to same when u build the database otherwise u will have to index again
3. Click **Build / Rebuild selected index** to create the index.
4. Choose a query (a file from the dataset, an upload, or a text description) and click **Search**.
