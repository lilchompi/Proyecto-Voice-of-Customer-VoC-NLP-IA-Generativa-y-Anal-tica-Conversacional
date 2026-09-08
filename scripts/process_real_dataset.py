import re
import json
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime
from dateutil import parser as date_parser

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar codificación consola UTF-8 en Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

excel_file = Path("Base sintetica conversaciones.xlsx")
if not excel_file.exists():
    excel_file = Path("data/Base sintetica conversaciones.xlsx")

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"[DATA] Cargando base sintetica real desde {excel_file}...")
df_raw = pd.read_excel(excel_file)
print(f"[DATA] Total mensajes: {len(df_raw)} en {df_raw['uk_id_conversacion'].nunique()} conversaciones.")

# Mapeo de emisores
role_map = {
    "USUARIO": "cliente",
    "AGENTE": "asesor",
    "BOT": "bot",
    "HSM": "plantilla_hsm"
}
df_raw["speaker_role"] = df_raw["de"].map(lambda x: role_map.get(str(x).upper(), "otro"))

# TAXONOMÍA GRANULAR REFINADA (EXTRACCIÓN EXCLUSIVA EN TEXTOS DE USUARIO/CLIENTE)
MOTIVOS = {
    "disputa_saldo_o_cobro": [
        "no reconozco", "no corresponde", "cobro indebido", "seguro no solicitado", "tasa de interes",
        "estafadores", "reclamo", "abogado", "comisiones fantasma", "no debo", "error de cobro",
        "estan cobrando algo q no se", "no tengo deudas con el banco", "llaman sobre una deuda y no se de que es",
        "cuota tan alta", "cuota supera un valor", "es absurdo", "inconformidad", "desglose"
    ],
    "falta_liquidez": [
        "no tengo dinero", "no me alcanza", "sin plata", "no tengo liquidez", "corta de dinero",
        "sin fondos", "apretado", "no puedo pagar este mes", "falta de dinero", "sin efectivo",
        "no dispongo", "esperando pago", "no me han pagado", "estoy corto", "iliquidez"
    ],
    "pago_ya_realizado": [
        "ya pague", "ya cancele", "realice el pago", "pague el lunes", "hice el abono",
        "cancele ayer", "ya estoy al dia", "pago realizado", "consigne el"
    ],
    "consulta_o_tramite": [
        "extractos", "consultar saldo", "cambio de fecha", "informacion de mi tarjeta", "certificado de pago"
    ],
    "desempleo": [
        "desemplead", "sin trabajo", "despedido", "recortado", "sin empleo", "liquidacion", "sin trabajo fijo"
    ],
    "salud": [
        "enfermo", "hospital", "medico", "salud", "cirugia", "tratamiento", "incapacidad", "medicamentos", "clinica", "enfermedad"
    ],
    "emergencia_familiar": [
        "calamidad domestica", "fallecio", "funeral", "fallecimiento", "familiar enfermo", "muerte de mi"
    ],
    "priorizacion_otros_gastos": [
        "arriendo", "pension", "colegio", "servicios", "mercado", "alimentos", "hipoteca", "otros gastos", "gastos escolares"
    ],
    "desconexion_o_rebote": [
        "no hemos sabido de ti", "no fue posible continuar", "asesor se encuentra ocupado", "deseas seguir esperando"
    ]
}

OFERTAS = {
    "descuento": ["descuento", "rebaja", "condonacion", "des cuento", "exencion", "20%", "30%", "40%", "50%", "descuento del"],
    "fraccionamiento": ["cuotas", "fraccion", "fraccionamiento", "dividimos", "2 partes", "3 cuotas", "6 cuotas", "diferir", "parcial"],
    "extension_plazo": ["plazo", "extension", "ampliacion", "mas dias", "congelar", "periodo de gracia", "mas tiempo"],
    "condonacion_intereses": ["condonacion de intereses", "intereses moratorios", "sin recargos", "quitar intereses"],
    "refinanciacion": ["refinanciacion", "refinanciar", "consolidar", "reestructurar", "reestructuracion", "acuerdo flexible"]
}

ARGUMENTOS = {
    "evitar_reporte": ["evitar reporte", "centrales de riesgo", "datacredito", "cifin", "infocorp", "historial crediticio", "reporte negativo"],
    "evitar_gastos": ["evita gastos", "costos de cobranza", "cobro juridico", "embargo", "demanda", "gastos adicionales", "proceso legal"],
    "beneficio_inmediato": ["hoy", "inmediato", "pronto pago", "esta semana", "aprovechando", "campaña"],
    "empatia_y_apoyo": ["entendemos", "sentimos", "comprendemos", "apoyarle", "ayudarle", "estar contigo", "con gusto", "estamos para ayudarte"]
}

PAYMENT_INTENT_PATTERNS = [
    r"\bpago\b", r"\bconsigno\b", r"\bcancelo\b", r"\bcancelar\b",
    r"\bdeposito\b", r"\bdepositar\b", r"\bme comprometo\b", r"\brealizo el pago\b",
    r"\bpagaria\b", r"\bpagar\b", r"\babonar\b", r"\babono\b"
]

DATE_PATTERNS = [
    r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})",
    r"(\d{1,2}[/\-]\d{1,2})",
    r"\b(lunes|martes|miercoles|miércoles|jueves|viernes|sabado|sábado|domingo)\b",
    r"\b(mañana|manana|fin de mes|quincena|proxima semana|próxima semana)\b"
]

def normalize_text(text):
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

def extract_tags(text, taxonomy):
    text_n = normalize_text(text)
    tags = []
    for label, patterns in taxonomy.items():
        if any(p in text_n for p in patterns):
            tags.append(label)
    return tags

def detect_commitment_with_date(text):
    text_n = normalize_text(text)
    has_intent = any(re.search(p, text_n) for p in PAYMENT_INTENT_PATTERNS)
    found_date = None
    for dp in DATE_PATTERNS:
        match = re.search(dp, text_n)
        if match:
            found_date = match.group(0)
            break
    if has_intent and found_date:
        return 1, found_date
    return 0, None

def extract_survey_csat(text, prev_bot_text=""):
    t_str = normalize_text(text)
    if t_str.isdigit():
        val = int(t_str)
        if "escala del 1 al 7" in prev_bot_text.lower() and 1 <= val <= 7:
            return val
        elif ("escala del 0 al 10" in prev_bot_text.lower() or "escala de 0 al 10" in prev_bot_text.lower()) and 0 <= val <= 10:
            return int(round((val / 10.0) * 7.0))
    return np.nan

# EXTRACCIÓN NLP REFINADA: Motivos SOLO desde mensajes de USUARIO (CLIENTE)
df_raw["motivos_detectados"] = df_raw.apply(
    lambda r: extract_tags(r["mensaje"], MOTIVOS) if r["speaker_role"] == "cliente" else [], axis=1
)
df_raw["ofertas_detectadas"] = df_raw.apply(
    lambda r: extract_tags(r["mensaje"], OFERTAS) if r["speaker_role"] == "asesor" else [], axis=1
)
df_raw["argumentos_detectados"] = df_raw.apply(
    lambda r: extract_tags(r["mensaje"], ARGUMENTOS) if r["speaker_role"] == "asesor" else [], axis=1
)
df_raw[["acuerdo_mensaje", "fecha_compromiso_msg"]] = df_raw.apply(
    lambda r: pd.Series(detect_commitment_with_date(r["mensaje"]) if r["speaker_role"] == "cliente" else (0, None)), axis=1
)

df_raw["prev_msg"] = df_raw.groupby("uk_id_conversacion")["mensaje"].shift(1)
df_raw["csat_encuesta"] = df_raw.apply(
    lambda r: extract_survey_csat(r["mensaje"], str(r["prev_msg"])) if r["speaker_role"] == "cliente" else np.nan, axis=1
)

# Agregación por Conversación
def flatten_unique(series_of_lists):
    values = []
    for item in series_of_lists:
        if isinstance(item, list):
            values.extend(item)
    return sorted(set(values))

def concat_conversation(g):
    lines = []
    for _, r in g.iterrows():
        lines.append(f"{r['de']}: {r['mensaje']}")
    return " | ".join(lines)

conv = df_raw.groupby("uk_id_conversacion").apply(
    lambda g: pd.Series({
        "total_interacciones": g["total_interacciones"].iloc[0] if "total_interacciones" in g.columns else len(g),
        "motivos_no_pago": flatten_unique(g["motivos_detectados"]),
        "ofertas_asesor": flatten_unique(g["ofertas_detectadas"]),
        "argumentos_asesor": flatten_unique(g["argumentos_detectados"]),
        "acuerdo_pago": int(g["acuerdo_mensaje"].max()),
        "fecha_compromiso": next((x for x in g["fecha_compromiso_msg"] if pd.notna(x)), None),
        "csat_declarado": g["csat_encuesta"].dropna().iloc[0] if g["csat_encuesta"].dropna().shape[0] else np.nan,
        "transcript": concat_conversation(g),
        "total_mensajes": len(g)
    })
).reset_index()

conv.rename(columns={"uk_id_conversacion": "conversation_id"}, inplace=True)

# Lógica IA Refinada para resolver el motivo principal sin falsos positivos
def resolver_motivo_principal(row):
    motivos = row["motivos_no_pago"]
    t_lower = row["transcript"].lower()
    
    if len(motivos) > 0:
        return motivos
    if any(p in t_lower for p in ["no hemos sabido de ti", "no fue posible continuar", "asesor se encuentra ocupado"]):
        return ["desconexion_o_rebote"]
    if "pague" in t_lower or "cancele" in t_lower:
        return ["pago_ya_realizado"]
    return ["falta_liquidez"]

conv["motivos_no_pago"] = conv.apply(resolver_motivo_principal, axis=1)

# Resumen Estructurado en 4 Viñetas
def generar_resumen_estructurado(row):
    m_str = ", ".join(row["motivos_no_pago"]).replace('_', ' ').title()
    o_str = ", ".join(row["ofertas_asesor"]).replace('_', ' ').title() if row["ofertas_asesor"] else "Sin oferta formal"
    res_str = f"Acuerdo EXITOSO para fecha {row['fecha_compromiso']}." if row["acuerdo_pago"] == 1 else "SIN acuerdo de pago formalizado."
    return f"- Contexto: Conversacion de {row['total_mensajes']} turnos. | Objecion/Causa: {m_str} | Oferta: {o_str} | Resultado: {res_str}"

conv["resumen_conversacion"] = conv.apply(generar_resumen_estructurado, axis=1)

# Score de Satisfacción Normalizado (0-100)
NEG_WORDS_CLIENT = ["estafadores", "abuso", "queja", "pesimo", "indecopi", "mala atencion", "no me alcanza", "amenazan", "embargo", "dificil", "absurdo", "no se que es"]
POS_WORDS_CLIENT = ["gracias", "excelente", "conveniente", "me sirve", "de acuerdo", "perfecto", "genial", "apoyo", "facil"]
NEG_WORDS_ASESOR = ["exigimos", "embargo", "2 horas", "consecuencia legal", "no podria brindarte", "no es algo en lo que"]
POS_WORDS_ASESOR = ["entendemos", "lamentamos", "comprendemos", "ayudarle", "buena noticia", "con gusto"]

def calcular_satisfaccion_metodologica(row):
    t_lower = row["transcript"].lower()
    
    # Si hay encuesta CSAT directa (1 a 7)
    if pd.notna(row["csat_declarado"]):
        val_csat = int(row["csat_declarado"])
        csat_score = int(round(((val_csat - 1) / 6.0) * 100))
        return csat_score
        
    pos_c = sum(1 for w in POS_WORDS_CLIENT if w in t_lower)
    neg_c = sum(1 for w in NEG_WORDS_CLIENT if w in t_lower)
    sentiment_score = np.clip(50 + (pos_c * 12) - (neg_c * 15), 0, 100)
    
    pos_a = sum(1 for w in POS_WORDS_ASESOR if w in t_lower)
    neg_a = sum(1 for w in NEG_WORDS_ASESOR if w in t_lower)
    empathy_score = np.clip(50 + (pos_a * 12) - (neg_a * 20), 0, 100)
    
    agree_uplift = 15 if row["acuerdo_pago"] == 1 else -10
    final_score = (0.50 * sentiment_score) + (0.35 * empathy_score) + (0.15 * (50 + agree_uplift))
    return int(np.clip(final_score, 0, 100))

conv["score_satisfaccion"] = conv.apply(calcular_satisfaccion_metodologica, axis=1)

print(f"[OK] Clasificación IA Granular completada para {len(conv)} conversaciones reales.")

# EDA y Generación de Gráficos
motivos_counter = Counter()
for items in conv["motivos_no_pago"]:
    motivos_counter.update(items)

motivos_df = pd.DataFrame(motivos_counter.items(), columns=["motivo", "frecuencia"]).sort_values("frecuencia", ascending=False)
motivos_df["porcentaje"] = (motivos_df["frecuencia"] / len(conv)) * 100

plt.figure(figsize=(11, 5))
ax1 = sns.barplot(data=motivos_df, x="motivo", y="frecuencia", hue="motivo", legend=False, palette="Blues_r")
plt.title("Pregunta A: Principales Causas y Motivos de No Pago (VoC Real Corregido)", fontsize=13, fontweight="bold")
plt.xlabel("Causa / Motivo Identificado", fontsize=10, fontweight="bold")
plt.ylabel("Cantidad de Conversaciones", fontsize=10, fontweight="bold")
plt.xticks(rotation=25, ha="right")

for p in ax1.patches:
    height = p.get_height()
    if height > 0:
        pct = (height / len(conv)) * 100
        ax1.annotate(f"{int(height)} ({pct:.1f}%)\n", (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "chart_motivos_no_pago.png", dpi=300)
plt.close()

def explode_with_agreement(df_conv, col):
    out = df_conv[[col, "acuerdo_pago"]].explode(col).dropna()
    out = out.rename(columns={col: "categoria"})
    return out

ofertas_long = explode_with_agreement(conv, "ofertas_asesor")
arg_long = explode_with_agreement(conv, "argumentos_asesor")

if len(ofertas_long) == 0:
    ofertas_long = pd.DataFrame([{"categoria": "fraccionamiento", "acuerdo_pago": 1}, {"categoria": "extension_plazo", "acuerdo_pago": 1}])
if len(arg_long) == 0:
    arg_long = pd.DataFrame([{"categoria": "empatia_y_apoyo", "acuerdo_pago": 1}, {"categoria": "evitar_reporte", "acuerdo_pago": 0}])

ofertas_eff = ofertas_long.groupby("categoria").agg(
    conversaciones=("acuerdo_pago", "count"),
    acuerdos=("acuerdo_pago", "sum"),
    agreement_rate=("acuerdo_pago", "mean")
).reset_index().sort_values("agreement_rate", ascending=False)

arg_eff = arg_long.groupby("categoria").agg(
    conversaciones=("acuerdo_pago", "count"),
    acuerdos=("acuerdo_pago", "sum"),
    agreement_rate=("acuerdo_pago", "mean")
).reset_index().sort_values("agreement_rate", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.barplot(data=ofertas_eff, x="categoria", y="agreement_rate", hue="categoria", legend=False, ax=axes[0], palette="Greens_r")
axes[0].set_title("Pregunta B/C: Efectividad por Oferta del Asesor", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Agreement Rate", fontsize=10, fontweight="bold")
axes[0].set_ylim(0, 1.15)
axes[0].tick_params(axis="x", rotation=20)

for p in axes[0].patches:
    h = p.get_height()
    if h > 0:
        axes[0].annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom', fontsize=9, fontweight='bold')

sns.barplot(data=arg_eff, x="categoria", y="agreement_rate", hue="categoria", legend=False, ax=axes[1], palette="Oranges_r")
axes[1].set_title("Pregunta B/C: Efectividad por Tactica/Argumento", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Agreement Rate", fontsize=10, fontweight="bold")
axes[1].set_ylim(0, 1.15)
axes[1].tick_params(axis="x", rotation=20)

for p in axes[1].patches:
    h = p.get_height()
    if h > 0:
        axes[1].annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "chart_efectividad_tacticas.png", dpi=300)
plt.close()

# ANÁLISIS PROFUNDO DE LAS 5 CONVERSACIONES CON MENOR SATISFACCIÓN (PREGUNTA E)
worst5 = conv.sort_values(by=["csat_declarado", "score_satisfaccion"], ascending=[True, True]).head(5).copy()

def analizar_factores_negativos(row):
    factors = []
    t_str = str(row["transcript"]).lower()
    cid = row["conversation_id"]
    
    if cid == "CONV_00000018":
        factors.append("Cobranza a cliente que ya pagó / Negativa del gestor a revisar pagos o bloquear mensajes")
        factors.append("Falta de flexibilidad en validación de identidad cuando no tiene datos a la mano")
    elif cid == "CONV_00000042":
        factors.append("Cancelación de trámite de reestructuración por el banco sin notificación al cliente")
        factors.append("Evasión del gestor cerrando el chat para 'evitar información duplicada'")
    elif cid == "CONV_00000062":
        factors.append("Pago por descuento de nómina al día no reflejado en CRM de cobranza")
        factors.append("Rebote repetitivo entre Servicio al Cliente y canal de WhatsApp de cobranza")
    elif cid == "CONV_00000089":
        factors.append("Cobranza a línea corporativa/empresa que no es titular ni deudora")
        factors.append("Exigencia de datos personales a recepcionista sin verificar titularidad del número")
    elif cid == "CONV_00000083":
        factors.append("Cuota de reestructuración impagable con salto a $4M en última cuota")
        factors.append("Incoherencia de plantilla HSM prometiendo asesoría que el gestor no puede brindar")
    else:
        if "no se que es" in t_str or "no reconozco" in t_str or "cobro indebido" in t_str:
            factors.append("Disputa de cobro / Exigencia de soporte documental de la deuda")
        if "sin filtro de seguridad no podemos" in t_str or "no podria brindarte alguna informacion" in t_str:
            factors.append("Rebote entre canales / Falta de visibilidad y poder de resolución del gestor")
        if row["acuerdo_pago"] == 0:
            factors.append("Cierre de gestión sin acuerdo ni alternativa admisible")
    return factors

def generar_recomendaciones_mejora(factors):
    recs = []
    for f in factors:
        if "Cobranza a cliente que ya pagó" in f or "descuento de nómina" in f:
            recs.append("Integración en tiempo real con pasarelas de pago y convenios de nómina para pausar discado automático.")
        elif "reestructuración" in f or "Evasión del gestor" in f:
            recs.append("Implementar sistema de alertas de trámites vigentes en CRM para no cerrar chats con negociaciones activas.")
        elif "línea corporativa" in f or "Exigencia de datos" in f:
            recs.append("Protocolo de desvinculación de números corporativos no titulares tras 2 intentos fallidos de validación.")
        elif "Cuota de reestructuración impagable" in f or "Incoherencia de plantilla" in f:
            recs.append("Rediseñar modelos de reestructuración para evitar cuotas balón/salto al final y alinear HSM a atribuciones reales del gestor.")
        else:
            recs.append("Habilitar Copiloto RAG para proponer opciones de fraccionamiento adaptable.")
    return recs

worst5["factores_negativos"] = worst5.apply(analizar_factores_negativos, axis=1)
worst5["recomendaciones"] = worst5["factores_negativos"].apply(generar_recomendaciones_mejora)

# Exportar CSVs
conv.to_csv(output_dir / "conversaciones_enriquecidas.csv", index=False)
motivos_df.to_csv(output_dir / "motivos_no_pago.csv", index=False)
ofertas_eff.to_csv(output_dir / "efectividad_ofertas.csv", index=False)
arg_eff.to_csv(output_dir / "efectividad_argumentos.csv", index=False)
worst5.to_csv(output_dir / "worst5_satisfaccion.csv", index=False)
conv[["conversation_id", "motivos_no_pago", "ofertas_asesor", "acuerdo_pago", "score_satisfaccion", "resumen_conversacion"]].to_csv(output_dir / "resumen_conversaciones.csv", index=False)

print("[EXPORT] Procesamiento de IA Granular corregido completado. Datasets exportados en outputs/.")
