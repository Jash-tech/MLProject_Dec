import os
from pathlib import Path

project_file="mlproject"

list_of_files=[
    f"src/{project_file}/__init__.py",
    f"src/{project_file}/components/__init__.py",
    f"src/{project_file}/components/data_ingestion.py",
    f"src/{project_file}/components/data_transformation.py",
    f"src/{project_file}/components/model_trainer.py",
    f"src/{project_file}/components/model_monitoring.py",
    f"src/{project_file}/pipeline/__init__.py",
    f"src/{project_file}/pipeline/training_pipeline.py",
    f"src/{project_file}/pipeline/prediction_pipeline.py",
    f"src/{project_file}/logger.py",
    f"src/{project_file}/exception.py",
    f"src/{project_file}/utils.py",
    "setup.py",
    "requirements.txt",
    "README.md"
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir,filename=os.path.split(filepath)


    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        print(f"Creating Dir {filedir} for {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):

        with open(filepath,'w') as f:
            pass



