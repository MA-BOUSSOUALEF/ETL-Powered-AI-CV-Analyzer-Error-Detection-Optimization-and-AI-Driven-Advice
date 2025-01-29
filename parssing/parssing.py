import pandas as pd 
import re

text = "Nom: John Doe\nCompétences: Python, SQL"
name_match = re.search(r"Nom:\s*(.+)", text)
if name_match:
    name = name_match.group(1)
    print(f"Nom: {name}")