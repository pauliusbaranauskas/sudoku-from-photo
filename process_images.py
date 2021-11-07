import os
import pandas as pd

working_dir = os.getcwd()

files = os.listdir(f"{working_dir}/images")

out_data = []

for file in files:
    filepath = f"{working_dir}/images/{file}"
    # print(file)
    if ".dat" in file:
        sudoku = []
        with open(filepath, "r") as f:
            data = f.readlines()[2:]
            for line in data:
                line = line.strip()
                sudoku.append(line)
            sudoku.append(filepath.replace("dat", "jpg"))
            out_data.append(sudoku)

out_data = pd.DataFrame(
    out_data,
    columns=[
        "line_1",
        "line_2",
        "line_3",
        "line_4",
        "line_5",
        "line_6",
        "line_7",
        "line_8",
        "line_9",
        "filepath"
    ],
)

out_data.to_pickle(f"{working_dir}/data.pkl")
