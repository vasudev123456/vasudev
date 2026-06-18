!pip install fastapi uvicorn nest_asyncio gradio
from fastapi import FastAPI, Body
import nest_asyncio, uvicorn, threading, gradio as gr, requests
app, codes = FastAPI(), []
@app.get("/")
def home(): return {"info": "Indian Penal Code", "queries": ["0: Theft", "1: Bribery", "2: Total sections"]}
@app.post("/ipc")
def add(fapi: dict = Body(...)): codes.append(fapi); return {"Details": f"{fapi['IPC']}, {fapi['case']}, {fapi['punishment']}"}
@app.get("/{i}")
def get(i: int): return codes[i] if 0 <= i < len(codes) else {"error": "Invalid index"}
nest_asyncio.apply()
threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000)).start()
demo = gr.Blocks()
with demo:
    gr.Markdown("# IPC Chatbot")
    gr.Button("Show Info").click(lambda: requests.get("http://127.0.0.1:8000/").json(), outputs=gr.Textbox())
    with gr.Row():
        i, c, p = gr.Textbox(placeholder="IPC"), gr.Textbox(placeholder="Case"), gr.Textbox(placeholder="Punishment")
        gr.Button("Add IPC").click(lambda ipc, case, pun: requests.post("http://127.0.0.1:8000/ipc", json={"IPC": ipc, "case": case, "punishment": pun}).json(), inputs=[i, c, p], outputs=gr.Textbox())
    idx = gr.Number(label="Index")
    gr.Button("Get IPC").click(lambda index: requests.get(f"http://127.0.0.1:8000/{int(index)}").json(), inputs=idx, outputs=gr.Textbox())

demo.launch(share=True)
