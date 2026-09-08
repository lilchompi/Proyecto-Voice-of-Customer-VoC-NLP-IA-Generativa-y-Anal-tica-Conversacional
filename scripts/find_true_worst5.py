import pandas as pd
import numpy as np
import re
from pathlib import Path

excel_file = Path("Base sintetica conversaciones.xlsx")
if not excel_path if 'excel_path' in locals() else False:
    excel_file = Path("Base sintetica conversaciones.xlsx")

df_raw = pd.read_excel("Base sintetica conversaciones.xlsx")

# Filtrar mensajes del cliente (USUARIO)
df_user = df_raw[df_raw["de"].str.upper() == "USUARIO"].copy()

# Extraer CSAT explícito de encuestas en el bot
df_raw["prev_msg"] = df_raw.groupby("uk_id_conversacion")["mensaje"].shift(1)

def extract_csat(row):
    t_str = str(row["mensaje"]).strip()
    p_str = str(row["prev_msg"]).strip().lower()
    if t_str.isdigit():
        val = int(t_str)
        if "escala del 1 al 7" in p_str and 1 <= val <= 7:
            return val # Escala 1-7
        elif ("escala del 0 al 10" in p_str or "escala de 0 al 10" in p_str) and 0 <= val <= 10:
            return val # Escala 0-10
    return np.nan

df_raw["csat_val"] = df_raw.apply(extract_csat, axis=1)

# Encontrar conversaciones con encuestas CSAT bajas (1/7, 2/7, 0/10, 1/10, etc.)
csat_convs = df_raw.groupby("uk_id_conversacion")["csat_val"].min().dropna().reset_index()
worst_csat_ids = csat_convs.sort_values("csat_val", ascending=True).head(15)["uk_id_conversacion"].tolist()

print(f"Top conversaciones con CSAT explícito más bajo: {worst_csat_ids[:10]}")

for cid in worst_csat_ids[:5]:
    c_df = df_raw[df_raw["uk_id_conversacion"] == cid]
    print(f"\n=================== {cid} (CSAT Min: {csat_convs[csat_convs['uk_id_conversacion']==cid]['csat_val'].values[0]}) ===================")
    for _, r in c_df.iterrows():
        print(f"  [{r['de']}]: {r['mensaje']}")
