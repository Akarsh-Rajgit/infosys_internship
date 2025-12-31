import joblib

PIPELINE_PATH = r"C:\Users\AKARSH RAJ M H\OneDrive\Desktop\Akarsh_Infosys_internship\milestone 3\real-estate-app\backend\models\pipeline_v1.pkl"
pipeline = joblib.load(PIPELINE_PATH)

print("Pipeline steps:")
for name, step in pipeline.steps:
    print(f"- {name}: {type(step)}")
