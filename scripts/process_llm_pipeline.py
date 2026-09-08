"""
Pipeline NLP + IA Generativa REAL con Google Gemini
=====================================================
Usa el SDK nuevo 'google-genai' (no el deprecado 'google-generativeai').
Modelo: gemini-2.5-flash-lite  (gratuito, JSON mode nativo)

Flujo:
  1. Leer Excel -> agrupar mensajes por conversacion
  2. Enviar cada transcript completo a Gemini con JSON Mode
  3. Gemini responde JSON estructurado (motivos, ofertas, acuerdo, tono, etc.)
  4. Agregar resultados + EDA + exportar CSVs

Uso:
  1. Editar .env con tu GOOGLE_API_KEY (https://aistudio.google.com/app/apikey)
  2. python scripts/process_llm_pipeline.py
     --sample 10     (solo 10 convs para prueba rapida)
     --sample 50     (muestra representativa, ~2 min)
     sin --sample    (todas las 1,197 conversaciones, ~35-40 min)
"""

import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# ── SDK nuevo (google-genai) ─────────────────────────────────────────────────
from google import genai
from google.genai import types

# ── Configuracion UTF-8 en consola Windows ───────────────────────────────────
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Cargar API key ────────────────────────────────────────────────────────────
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "PEGA_TU_API_KEY_AQUI":
    print("[ERROR] No se encontro GOOGLE_API_KEY en el archivo .env")
    print("        Edita .env y pega tu API key de: https://aistudio.google.com/app/apikey")
    sys.exit(1)

client = genai.Client(api_key=GOOGLE_API_KEY)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# ── Paths ────────────────────────────────────────────────────────────────────
excel_file = Path("data/Base sintetica conversaciones.xlsx")
if not excel_file.exists():
    excel_file = Path("Base sintetica conversaciones.xlsx")

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# ── MODELO ────────────────────────────────────────────────────────────────────
MODEL_NAME = "gemini-2.5-flash-lite"

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Eres un analista experto en Voice of Customer (VoC) especializado
en operaciones de cobranza bancaria por WhatsApp en America Latina.

REGLAS CRITICAS:
1. motivos_no_pago SOLO de lo que el CLIENTE dijo explicitamente.
   NO inferas motivos de mensajes del ASESOR, BOT o HSM.
2. acuerdo_pago = 1 UNICAMENTE si el CLIENTE expresa intencion de pago Y
   menciona una fecha especifica (dia/mes o dia de la semana).
   Solo 'voy a pagar' sin fecha -> acuerdo_pago = 0.
3. Si cliente dice 'ya pague' o similar -> incluye pago_ya_realizado en motivos
   y pon requiere_escalamiento = true. NO insistir en cobro.
4. Disputa o cliente no reconoce deuda -> requiere_escalamiento = true.
5. Sin motivo explicito del cliente -> motivos_no_pago = ["desconexion_o_rebote"].
6. Responde UNICAMENTE con JSON valido. Sin texto extra. Sin markdown.

TAXONOMIA DE MOTIVOS (usa solo estas categorias):
  desconexion_o_rebote | falta_liquidez | pago_ya_realizado
  priorizacion_otros_gastos | desempleo | salud
  disputa_saldo_o_cobro | consulta_o_tramite | emergencia_familiar

TAXONOMIA DE OFERTAS (usa solo estas):
  descuento | fraccionamiento | extension_plazo
  condonacion_intereses | refinanciacion | derivacion_reclamos

TAXONOMIA DE ARGUMENTOS (usa solo estos):
  evitar_reporte | evitar_gastos | beneficio_inmediato | empatia_y_apoyo"""

# ── JSON SCHEMA ───────────────────────────────────────────────────────────────
OUTPUT_SCHEMA = {
    "motivos_no_pago": ["desconexion_o_rebote"],
    "submotivo": "descripcion con palabras del cliente",
    "ofertas_asesor": ["fraccionamiento"],
    "argumentos_asesor": ["evitar_reporte"],
    "acuerdo_pago": 0,
    "fecha_compromiso": None,
    "tono_cliente": "neutral",
    "tono_asesor": "neutral",
    "requiere_escalamiento": False,
    "resumen_conversacion": "resumen en 2-3 oraciones",
    "score_riesgo_cx": 5
}

FALLBACK = {
    "motivos_no_pago": ["desconexion_o_rebote"],
    "submotivo": "Error al procesar con LLM",
    "ofertas_asesor": [],
    "argumentos_asesor": [],
    "acuerdo_pago": 0,
    "fecha_compromiso": None,
    "tono_cliente": "neutral",
    "tono_asesor": "neutral",
    "requiere_escalamiento": False,
    "resumen_conversacion": "No se pudo analizar esta conversacion.",
    "score_riesgo_cx": 5
}


def analizar_conversacion_con_llm(conv_id: str, transcript_lines: list,
                                   max_retries: int = 3) -> dict:
    """Envia transcript a Gemini con JSON Mode. Retorna dict parseado."""
    transcript_text = "\n".join(transcript_lines)
    prompt = (
        f"Analiza la siguiente conversacion de WhatsApp de cobranza bancaria.\n"
        f"Conversation ID: {conv_id}\n\n"
        f"--- INICIO ---\n{transcript_text}\n--- FIN ---\n\n"
        f"Responde con un JSON que siga exactamente este schema de ejemplo:\n"
        f"{json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"IMPORTANTE: Responde SOLO con el JSON, sin texto adicional."
    )

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[SYSTEM_PROMPT, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            raw = response.text.strip()
            raw = re.sub(r'^```json\s*', '', raw)
            raw = re.sub(r'```$', '', raw).strip()
            return json.loads(raw)
        except Exception as e:
            wait = (2 ** attempt) * 2
            log.warning(f"  [conv={conv_id}] Intento {attempt+1} fallido: {e}. Esperando {wait}s...")
            time.sleep(wait)

    log.error(f"  [conv={conv_id}] Todos los intentos fallaron. Usando fallback.")
    return FALLBACK.copy()


def calcular_score_satisfaccion(row: pd.Series) -> int:
    """Score 0-100. CSAT declarado tiene prioridad; si no, usa score_riesgo_cx del LLM."""
    if pd.notna(row.get("csat_declarado")):
        val = int(row["csat_declarado"])
        return int(round(((val - 1) / 6.0) * 100))
    base = int(np.clip((10 - row.get("score_riesgo_cx", 5)) * 10, 0, 100))
    if row.get("acuerdo_pago", 0) == 1:
        base = min(base + 10, 100)
    if row.get("requiere_escalamiento", False):
        base = max(base - 20, 0)
    return base


def main():
    parser = argparse.ArgumentParser(description="Pipeline LLM VoC con Google Gemini")
    parser.add_argument("--sample", type=int, default=None,
                        help="Procesar solo N conversaciones (para pruebas)")
    parser.add_argument("--delay", type=float, default=1.5,
                        help="Segundos entre llamadas API (default: 1.5)")
    args = parser.parse_args()

    # ── 1. CARGAR DATOS ───────────────────────────────────────────────────────
    log.info(f"Cargando dataset desde {excel_file}...")
    df_raw = pd.read_excel(excel_file)
    log.info(f"Total mensajes: {len(df_raw)}, Conversaciones: {df_raw['uk_id_conversacion'].nunique()}")

    role_map = {"USUARIO": "CLIENTE", "AGENTE": "ASESOR", "BOT": "BOT", "HSM": "HSM"}
    df_raw["rol"] = df_raw["de"].map(lambda x: role_map.get(str(x).upper(), str(x).upper()))

    # ── 2. CSAT DE ENCUESTA (heuristica, no necesita LLM) ────────────────────
    df_raw["prev_msg"] = df_raw.groupby("uk_id_conversacion")["mensaje"].shift(1).fillna("")
    df_raw["csat_encuesta"] = np.nan
    for idx, row in df_raw.iterrows():
        if row["rol"] == "CLIENTE":
            t = str(row["mensaje"]).strip()
            prev = str(row["prev_msg"]).lower()
            if t.isdigit():
                val = int(t)
                if "escala del 1 al 7" in prev and 1 <= val <= 7:
                    df_raw.at[idx, "csat_encuesta"] = val
                elif ("escala del 0 al 10" in prev or "escala de 0 al 10" in prev) and 0 <= val <= 10:
                    df_raw.at[idx, "csat_encuesta"] = int(round((val / 10.0) * 7.0))

    # ── 3. AGRUPAR CONVERSACIONES ─────────────────────────────────────────────
    grouped = df_raw.groupby("uk_id_conversacion")
    all_ids = list(grouped.groups.keys())
    if args.sample:
        all_ids = all_ids[:args.sample]
        log.info(f"Modo muestra: procesando {args.sample} conversaciones.")

    # ── 4. LOOP PRINCIPAL: ENVIAR CADA CONV AL LLM ────────────────────────────
    results = []
    total = len(all_ids)
    log.info(f"Iniciando analisis LLM de {total} conversaciones | Modelo: {MODEL_NAME}")

    for i, conv_id in enumerate(all_ids, 1):
        g = grouped.get_group(conv_id)
        lines = [
            f"[{r['rol']}]: {str(r['mensaje']).strip()}"
            for _, r in g.iterrows()
            if str(r['mensaje']).strip().lower() not in ['nan', '']
        ]
        csat_vals = g["csat_encuesta"].dropna()
        csat_decl = float(csat_vals.iloc[0]) if len(csat_vals) > 0 else np.nan

        if i % 10 == 0 or i == 1:
            log.info(f"  [{i}/{total}] {conv_id} ({len(lines)} turnos)...")

        llm = analizar_conversacion_con_llm(conv_id, lines)

        results.append({
            "conversation_id":       conv_id,
            "total_interacciones":   g["total_interacciones"].iloc[0] if "total_interacciones" in g.columns else len(g),
            "total_mensajes":        len(g),
            "csat_declarado":        csat_decl,
            "motivos_no_pago":       llm.get("motivos_no_pago", []),
            "submotivo":             llm.get("submotivo", ""),
            "ofertas_asesor":        llm.get("ofertas_asesor", []),
            "argumentos_asesor":     llm.get("argumentos_asesor", []),
            "acuerdo_pago":          int(llm.get("acuerdo_pago", 0)),
            "fecha_compromiso":      llm.get("fecha_compromiso"),
            "tono_cliente":          llm.get("tono_cliente", "neutral"),
            "tono_asesor":           llm.get("tono_asesor", "neutral"),
            "requiere_escalamiento": bool(llm.get("requiere_escalamiento", False)),
            "resumen_conversacion":  llm.get("resumen_conversacion", ""),
            "score_riesgo_cx":       int(llm.get("score_riesgo_cx", 5)),
            "transcript":            "\n".join(lines),
        })

        if i % 100 == 0:
            pd.DataFrame(results).to_csv(output_dir / f"checkpoint_{i}.csv", index=False)
            log.info(f"  [CHECKPOINT] Guardado en outputs/checkpoint_{i}.csv")

        time.sleep(args.delay)

    # ── 5. DATAFRAME FINAL ────────────────────────────────────────────────────
    conv = pd.DataFrame(results)
    conv["score_satisfaccion"] = conv.apply(calcular_score_satisfaccion, axis=1)

    log.info(f"[OK] Procesamiento completado: {len(conv)} conversaciones")
    log.info(f"     Acuerdos:           {conv['acuerdo_pago'].sum()} ({conv['acuerdo_pago'].mean()*100:.1f}%)")
    log.info(f"     Escalamientos:      {conv['requiere_escalamiento'].sum()}")
    log.info(f"     Score CX promedio:  {conv['score_satisfaccion'].mean():.1f}/100")

    # ── 6. EDA: GRAFICO DE MOTIVOS ────────────────────────────────────────────
    motivos_counter = Counter()
    for items in conv["motivos_no_pago"]:
        if isinstance(items, list):
            motivos_counter.update(items)

    motivos_df = (
        pd.DataFrame(motivos_counter.items(), columns=["motivo", "frecuencia"])
        .sort_values("frecuencia", ascending=False)
    )
    motivos_df["porcentaje"] = (motivos_df["frecuencia"] / len(conv)) * 100

    plt.figure(figsize=(11, 5))
    ax1 = sns.barplot(data=motivos_df, x="motivo", y="frecuencia",
                      hue="motivo", legend=False, palette="Blues_r")
    plt.title("Motivos de No Pago — Clasificacion Semantica con Gemini LLM", fontsize=13, fontweight="bold")
    plt.xlabel("Motivo", fontsize=10, fontweight="bold")
    plt.ylabel("Cantidad de Conversaciones", fontsize=10, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    for p in ax1.patches:
        h = p.get_height()
        if h > 0:
            pct = (h / len(conv)) * 100
            ax1.annotate(f"{int(h)} ({pct:.1f}%)",
                         (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "chart_motivos_no_pago.png", dpi=300)
    plt.close()

    # ── 7. EDA: EFECTIVIDAD OFERTAS Y ARGUMENTOS ──────────────────────────────
    def explode_eff(col):
        out = conv[[col, "acuerdo_pago"]].explode(col).dropna()
        out = out[out[col].astype(str).str.len() > 0].rename(columns={col: "categoria"})
        return out.groupby("categoria").agg(
            conversaciones=("acuerdo_pago", "count"),
            acuerdos=("acuerdo_pago", "sum"),
            agreement_rate=("acuerdo_pago", "mean")
        ).reset_index().sort_values("agreement_rate", ascending=False)

    ofertas_eff = explode_eff("ofertas_asesor")
    arg_eff     = explode_eff("argumentos_asesor")

    if len(ofertas_eff) > 0 and len(arg_eff) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for ax, df_e, title, pal in [
            (axes[0], ofertas_eff, "Efectividad por Oferta (LLM)", "Greens_r"),
            (axes[1], arg_eff,    "Efectividad por Argumento (LLM)", "Oranges_r"),
        ]:
            sns.barplot(data=df_e, x="categoria", y="agreement_rate",
                        hue="categoria", legend=False, ax=ax, palette=pal)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Agreement Rate", fontsize=10)
            ax.tick_params(axis="x", rotation=20)
            for p in ax.patches:
                h = p.get_height()
                if h > 0:
                    ax.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "chart_efectividad_tacticas.png", dpi=300)
        plt.close()

    # ── 8. WORST 5 ────────────────────────────────────────────────────────────
    worst5 = conv.sort_values(
        by=["csat_declarado", "score_satisfaccion"], ascending=[True, True]
    ).head(5).copy()

    def recomendacion(row):
        recs = []
        motivos = row.get("motivos_no_pago", [])
        if "pago_ya_realizado" in motivos:
            recs.append("Integrar verificacion de pagos en tiempo real antes de continuar gestion.")
        if "disputa_saldo_o_cobro" in motivos:
            recs.append("Derivar a Mesa de Reclamos y suspender cobro preventivamente.")
        if row.get("requiere_escalamiento"):
            recs.append("Activar protocolo de escalamiento automatico al supervisor.")
        if "desconexion_o_rebote" in motivos:
            recs.append("Implementar protocolo anti-rebote: resolver directamente en canal actual.")
        if not recs:
            recs.append("Usar Copiloto RAG para proponer alternativas adaptadas al motivo.")
        return recs

    worst5["recomendaciones"] = worst5.apply(recomendacion, axis=1)

    # ── 9. EXPORTAR CSVs ──────────────────────────────────────────────────────
    conv.to_csv(output_dir / "conversaciones_enriquecidas.csv", index=False)
    motivos_df.to_csv(output_dir / "motivos_no_pago.csv", index=False)
    ofertas_eff.to_csv(output_dir / "efectividad_ofertas.csv", index=False)
    arg_eff.to_csv(output_dir / "efectividad_argumentos.csv", index=False)
    worst5.to_csv(output_dir / "worst5_satisfaccion.csv", index=False)
    conv[[
        "conversation_id", "motivos_no_pago", "ofertas_asesor", "argumentos_asesor",
        "acuerdo_pago", "score_satisfaccion", "tono_cliente", "tono_asesor",
        "requiere_escalamiento", "resumen_conversacion"
    ]].to_csv(output_dir / "resumen_conversaciones.csv", index=False)

    log.info("[OK] Todos los outputs exportados en outputs/")
    print("\n=== DISTRIBUCION FINAL DE MOTIVOS (LLM SEMANTICO) ===")
    print(motivos_df.to_string(index=False))
    print("\n=== TONOS DEL CLIENTE ===")
    print(conv["tono_cliente"].value_counts().to_string())


if __name__ == "__main__":
    main()
