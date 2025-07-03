from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/govern")
def govern(data: dict):
    return {"status": "success", "prompt": data.get("prompt", ""), "output": data.get("output", "")}
