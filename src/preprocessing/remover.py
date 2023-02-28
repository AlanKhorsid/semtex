import os

try:
    folder_path = "src/preprocessing/output"

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.endswith(".csv"):
            with open(file_path, "r") as f:
                print(file_path)
                text = f.read()
            with open(file_path, "w") as f:
                print(file_path)
                f.write(text.replace("\n", ""))
except Exception as e:
    print(e)
