from fastapi import FastAPI

from sample_project.routes import router

app = FastAPI(title="User Manager")
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
