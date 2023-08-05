# chatbot
App web desarrollada utili


## Running Locally

**1. Clone Repo**

```bash
git clone https://github.com/JGPOAE99/chatbot.git
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Provide API Keys**

Create a `.env` local file in the root of the repo with your API Keys:

```
QDRANT_API_KEY=
QDRANT_HOST=
QDRANT_COLLECTION_NAME=
OPENAI_API_KEY=
```

**4. Create vector store**
After uploading some `*.pdf` documents to your folders `/data/*`, call the function `load_documents(data_path: string)` from `data_ingest_qdrant.py` to create the vector store on [Qdrant Cloud](https://qdrant.tech/). (You will need to create a Free cluster before doing this)


**5. Run app**
```bash
streamlit run app.py
```