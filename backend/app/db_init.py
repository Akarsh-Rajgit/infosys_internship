# backend/app/init_db.py
from .db import engine
from . import models

print("🚀 Creating database tables...")
models.Base.metadata.create_all(bind=engine)
print("✅ Tables created successfully!")
