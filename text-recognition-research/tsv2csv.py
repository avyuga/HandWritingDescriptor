import pandas as pd

for split in ["test"]:
    csv_table = pd.read_table(f"../data/cyrillic-handwriting-dataset/{split}.tsv", sep='\t')
    csv_table.columns = ["filename", "words"]
    csv_table.to_csv(f'../data/cyrillic-handwriting-dataset/{split}/labels.csv',
                     index=False
                    )
