"""
Reemplaza las celdas de heuristicas regex del notebook principal
por el pipeline LLM real con Google Gemini.
Edita directamente Prueba_Tecnica_VoC_NLP_IA_Generativa.ipynb
"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

NB_PATH = 'Prueba_Tecnica_VoC_NLP_IA_Generativa.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# ── Nuevas celdas que van a REEMPLAZAR las secciones 3 y 4 heurísticas ────────

CELL_MD_SEC3 = {
    "cell_type": "markdown",
    "id": "sec_03_llm_md",
    "metadata": {},
    "source": [
        "## 3. Pipeline de IA Generativa con Google Gemini — Clasificación Semántica Real\n",
        "\n",
        "Este módulo **reemplaza completamente** las expresiones regulares y diccionarios heurísticos\n",
        "por llamadas reales a **Google Gemini Flash** usando **JSON Mode nativo**.\n",
        "El LLM comprende el contexto semántico completo de cada conversación:\n",
        "\n",
        "| Limitación del regex | Solución con LLM |\n",
        "|---|---|\n",
        "| `'ya pagué pero sigue en mora'` → falso positivo `pago_ya_realizado` | LLM detecta contexto real → `disputa_saldo_o_cobro` |\n",
        "| `'ando en la olla'` → sin clasificar | LLM entiende modismo → `falta_liquidez` |\n",
        "| Acuerdo sin fecha marcado como acuerdo | LLM aplica regla: intención + fecha = acuerdo |\n",
        "| Sin detección de tono emocional | LLM clasifica: frustrado / colaborativo / evasivo |"
    ]
}

CELL_CODE_SEC3 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "code_03_llm_setup",
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, re, time, json as _json\n",
        "import numpy as np\n",
        "from dotenv import load_dotenv\n",
        "import google.generativeai as genai\n",
        "\n",
        "# Cargar API key desde .env\n",
        "load_dotenv()\n",
        "GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')\n",
        "assert GOOGLE_API_KEY and GOOGLE_API_KEY != 'PEGA_TU_API_KEY_AQUI', \\\n",
        "    'ERROR: Edita el archivo .env y pega tu API key de https://aistudio.google.com/app/apikey'\n",
        "genai.configure(api_key=GOOGLE_API_KEY)\n",
        "\n",
        "MODEL_NAME = 'gemini-2.5-flash-lite-preview-06-17'\n",
        "llm_model = genai.GenerativeModel(MODEL_NAME)\n",
        "print(f'[OK] Conectado a Gemini: {MODEL_NAME}')\n",
        "\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "# SYSTEM PROMPT CON REGLAS DE NEGOCIO\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "SYSTEM_PROMPT = '''Eres un analista experto en Voice of Customer (VoC) especializado\n",
        "en operaciones de cobranza bancaria por WhatsApp en America Latina.\n",
        "\n",
        "REGLAS CRITICAS:\n",
        "1. motivos_no_pago SOLO de lo que el CLIENTE dijo explicitamente.\n",
        "   NO inferas motivos de mensajes del ASESOR, BOT o HSM.\n",
        "2. acuerdo_pago = 1 UNICAMENTE si el CLIENTE expresa intencion de pago Y\n",
        "   menciona una fecha especifica (dia/mes o dia de la semana).\n",
        "   Solo 'voy a pagar' sin fecha -> acuerdo_pago = 0.\n",
        "3. Si cliente dice 'ya pague' o similar -> motivos incluye pago_ya_realizado\n",
        "   y requiere_escalamiento = true. NO insistir en cobro.\n",
        "4. Disputa o cliente no reconoce deuda -> requiere_escalamiento = true.\n",
        "5. Sin motivo explicito del cliente -> motivos_no_pago = [desconexion_o_rebote].\n",
        "6. Responde UNICAMENTE con JSON valido. Sin texto extra. Sin markdown.\n",
        "\n",
        "TAXONOMIA DE MOTIVOS (usa solo estas categorias en la lista):\n",
        "  desconexion_o_rebote | falta_liquidez | pago_ya_realizado\n",
        "  priorizacion_otros_gastos | desempleo | salud\n",
        "  disputa_saldo_o_cobro | consulta_o_tramite | emergencia_familiar\n",
        "\n",
        "TAXONOMIA DE OFERTAS (usa solo estas):\n",
        "  descuento | fraccionamiento | extension_plazo\n",
        "  condonacion_intereses | refinanciacion | derivacion_reclamos\n",
        "\n",
        "TAXONOMIA DE ARGUMENTOS (usa solo estos):\n",
        "  evitar_reporte | evitar_gastos | beneficio_inmediato | empatia_y_apoyo\n",
        "'''\n",
        "\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "# JSON SCHEMA DE SALIDA ESTRUCTURADA\n",
        "# ─────────────────────────────────────────────────────────────────────────────\n",
        "OUTPUT_SCHEMA = {\n",
        "    '$schema': 'http://json-schema.org/draft-07/schema',\n",
        "    'title': 'AnalisisConversacionCobranza',\n",
        "    'type': 'object',\n",
        "    'required': ['motivos_no_pago', 'acuerdo_pago', 'requiere_escalamiento',\n",
        "                 'resumen_conversacion', 'score_riesgo_cx'],\n",
        "    'properties': {\n",
        "        'motivos_no_pago': {\n",
        "            'type': 'array', 'items': {'type': 'string'},\n",
        "            'description': 'Motivos expresados EXPLICITAMENTE por el CLIENTE'\n",
        "        },\n",
        "        'submotivo': {'type': 'string', 'description': 'Descripcion con las palabras del cliente'},\n",
        "        'ofertas_asesor': {'type': 'array', 'items': {'type': 'string'}},\n",
        "        'argumentos_asesor': {'type': 'array', 'items': {'type': 'string'}},\n",
        "        'acuerdo_pago': {\n",
        "            'type': 'integer', 'enum': [0, 1],\n",
        "            'description': '1 SOLO si cliente da intencion + fecha especifica'\n",
        "        },\n",
        "        'fecha_compromiso': {'type': ['string', 'null']},\n",
        "        'tono_cliente': {\n",
        "            'type': 'string',\n",
        "            'enum': ['colaborativo', 'neutral', 'frustrado', 'indignado', 'evasivo']\n",
        "        },\n",
        "        'tono_asesor': {\n",
        "            'type': 'string',\n",
        "            'enum': ['empatico', 'neutral', 'presionador', 'ineficaz']\n",
        "        },\n",
        "        'requiere_escalamiento': {\n",
        "            'type': 'boolean',\n",
        "            'description': 'true si hay disputa, pago ya realizado o cobro erroneo'\n",
        "        },\n",
        "        'resumen_conversacion': {\n",
        "            'type': 'string',\n",
        "            'description': 'Resumen en 2-3 oraciones: contexto | objecion | resultado'\n",
        "        },\n",
        "        'score_riesgo_cx': {\n",
        "            'type': 'integer', 'minimum': 0, 'maximum': 10,\n",
        "            'description': '10=muy alto riesgo de mala experiencia, 0=sin riesgo'\n",
        "        }\n",
        "    }\n",
        "}\n",
        "\n",
        "print('[OK] System Prompt y JSON Schema configurados.')\n",
        "print(f'     {len(OUTPUT_SCHEMA[\"properties\"])} campos de salida estructurada.')\n",
        "print()\n",
        "print('=== JSON SCHEMA DE SALIDA ===')\n",
        "print(_json.dumps(OUTPUT_SCHEMA, indent=2, ensure_ascii=False))"
    ]
}

CELL_MD_SEC4 = {
    "cell_type": "markdown",
    "id": "sec_04_llm_md",
    "metadata": {},
    "source": [
        "## 4. Ejecución del Pipeline LLM Batch (Gemini analiza cada conversación)\n",
        "\n",
        "> **Configuración de `SAMPLE_N`**:\n",
        "> - `SAMPLE_N = 10` → prueba rápida (~20 seg)\n",
        "> - `SAMPLE_N = 50` → validación representativa (~2 min)\n",
        "> - `SAMPLE_N = None` → todas las 1,197 conversaciones (~35-40 min en Free Tier)\n",
        ">\n",
        "> El pipeline guarda un **checkpoint CSV cada 100 conversaciones** para no perder avance."
    ]
}

CELL_CODE_SEC4 = {
    "cell_type": "code",
    "execution_count": None,
    "id": "code_04_llm_pipeline",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─── PARÁMETROS ───────────────────────────────────────────────────────────────\n",
        "SAMPLE_N = None  # None = todas las conversaciones | int = solo N para prueba\n",
        "DELAY_S  = 1.5   # segundos entre llamadas (respeta rate limits del Free Tier)\n",
        "\n",
        "# ─── FUNCIÓN DE LLAMADA AL LLM ────────────────────────────────────────────────\n",
        "def analizar_conv_llm(conv_id, transcript_lines, max_retries=3):\n",
        "    '''Envia transcript completo a Gemini con JSON Mode. Retorna dict parseado.'''\n",
        "    transcript_text = '\\n'.join(transcript_lines)\n",
        "    user_prompt = (\n",
        "        f'Analiza la siguiente conversacion de WhatsApp de cobranza bancaria.\\n'\n",
        "        f'Conversation ID: {conv_id}\\n\\n'\n",
        "        f'--- INICIO DE CONVERSACION ---\\n{transcript_text}\\n--- FIN ---\\n\\n'\n",
        "        f'Responde con un JSON siguiendo este schema:\\n'\n",
        "        f'{_json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}\\n\\n'\n",
        "        f'IMPORTANTE: Responde SOLO con el JSON, sin texto adicional ni markdown.'\n",
        "    )\n",
        "    for attempt in range(max_retries):\n",
        "        try:\n",
        "            resp = llm_model.generate_content(\n",
        "                [SYSTEM_PROMPT, user_prompt],\n",
        "                generation_config=genai.GenerationConfig(\n",
        "                    temperature=0.1,\n",
        "                    response_mime_type='application/json'  # JSON Mode nativo de Gemini\n",
        "                )\n",
        "            )\n",
        "            raw = resp.text.strip()\n",
        "            # Limpiar markdown si viene envuelto\n",
        "            raw = re.sub(r'^```json\\s*', '', raw)\n",
        "            raw = re.sub(r'```$', '', raw).strip()\n",
        "            return _json.loads(raw)\n",
        "        except Exception as e:\n",
        "            wait = (2 ** attempt) * 2  # backoff: 2s, 4s, 8s\n",
        "            print(f'  [WARN] {conv_id} intento {attempt+1} fallido: {e}. Esperando {wait}s...')\n",
        "            time.sleep(wait)\n",
        "    # Fallback si todos los intentos fallan\n",
        "    return {\n",
        "        'motivos_no_pago': ['desconexion_o_rebote'], 'submotivo': 'Error LLM',\n",
        "        'ofertas_asesor': [], 'argumentos_asesor': [],\n",
        "        'acuerdo_pago': 0, 'fecha_compromiso': None,\n",
        "        'tono_cliente': 'neutral', 'tono_asesor': 'neutral',\n",
        "        'requiere_escalamiento': False,\n",
        "        'resumen_conversacion': 'No se pudo analizar esta conversacion.',\n",
        "        'score_riesgo_cx': 5\n",
        "    }\n",
        "\n",
        "# ─── PREPARAR DATOS ───────────────────────────────────────────────────────────\n",
        "role_map = {'USUARIO': 'CLIENTE', 'AGENTE': 'ASESOR', 'BOT': 'BOT', 'HSM': 'HSM'}\n",
        "df_raw['rol'] = df_raw['de'].map(lambda x: role_map.get(str(x).upper(), str(x).upper()))\n",
        "\n",
        "# CSAT de encuesta — detección heurística pura (no necesita LLM)\n",
        "df_raw['prev_msg'] = df_raw.groupby('uk_id_conversacion')['mensaje'].shift(1).fillna('')\n",
        "df_raw['csat_encuesta'] = np.nan\n",
        "for idx, row in df_raw.iterrows():\n",
        "    if row['rol'] == 'CLIENTE':\n",
        "        t = str(row['mensaje']).strip()\n",
        "        prev = str(row['prev_msg']).lower()\n",
        "        if t.isdigit():\n",
        "            val = int(t)\n",
        "            if 'escala del 1 al 7' in prev and 1 <= val <= 7:\n",
        "                df_raw.at[idx, 'csat_encuesta'] = val\n",
        "            elif ('escala del 0 al 10' in prev or 'escala de 0 al 10' in prev) and 0 <= val <= 10:\n",
        "                df_raw.at[idx, 'csat_encuesta'] = int(round((val / 10.0) * 7.0))\n",
        "\n",
        "grouped    = df_raw.groupby('uk_id_conversacion')\n",
        "all_ids    = list(grouped.groups.keys())\n",
        "if SAMPLE_N:\n",
        "    all_ids = all_ids[:SAMPLE_N]\n",
        "    print(f'[INFO] Modo muestra: {SAMPLE_N} conversaciones')\n",
        "\n",
        "# ─── LOOP PRINCIPAL ───────────────────────────────────────────────────────────\n",
        "results = []\n",
        "total   = len(all_ids)\n",
        "print(f'Iniciando analisis LLM de {total} conversaciones con {MODEL_NAME}...')\n",
        "\n",
        "output_dir = Path('outputs')\n",
        "output_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "for i, conv_id in enumerate(all_ids, 1):\n",
        "    g = grouped.get_group(conv_id)\n",
        "    # Construir transcript formateado para el LLM\n",
        "    lines = [\n",
        "        f\"[{r['rol']}]: {str(r['mensaje']).strip()}\"\n",
        "        for _, r in g.iterrows()\n",
        "        if str(r['mensaje']).strip().lower() not in ['nan', '']\n",
        "    ]\n",
        "    csat_vals = g['csat_encuesta'].dropna()\n",
        "    csat_decl = float(csat_vals.iloc[0]) if len(csat_vals) > 0 else np.nan\n",
        "\n",
        "    if i % 10 == 0 or i == 1:\n",
        "        print(f'  [{i}/{total}] {conv_id} ({len(lines)} turnos)...')\n",
        "\n",
        "    llm = analizar_conv_llm(conv_id, lines)\n",
        "\n",
        "    results.append({\n",
        "        'conversation_id':      conv_id,\n",
        "        'total_interacciones':  g['total_interacciones'].iloc[0] if 'total_interacciones' in g.columns else len(g),\n",
        "        'total_mensajes':       len(g),\n",
        "        'csat_declarado':       csat_decl,\n",
        "        'motivos_no_pago':      llm.get('motivos_no_pago', []),\n",
        "        'submotivo':            llm.get('submotivo', ''),\n",
        "        'ofertas_asesor':       llm.get('ofertas_asesor', []),\n",
        "        'argumentos_asesor':    llm.get('argumentos_asesor', []),\n",
        "        'acuerdo_pago':         int(llm.get('acuerdo_pago', 0)),\n",
        "        'fecha_compromiso':     llm.get('fecha_compromiso'),\n",
        "        'tono_cliente':         llm.get('tono_cliente', 'neutral'),\n",
        "        'tono_asesor':          llm.get('tono_asesor', 'neutral'),\n",
        "        'requiere_escalamiento': bool(llm.get('requiere_escalamiento', False)),\n",
        "        'resumen_conversacion': llm.get('resumen_conversacion', ''),\n",
        "        'score_riesgo_cx':      int(llm.get('score_riesgo_cx', 5)),\n",
        "        'transcript':           '\\n'.join(lines),\n",
        "    })\n",
        "\n",
        "    # Checkpoint cada 100 conversaciones\n",
        "    if i % 100 == 0:\n",
        "        pd.DataFrame(results).to_csv(output_dir / f'checkpoint_{i}.csv', index=False)\n",
        "        print(f'  [CHECKPOINT] Guardado en outputs/checkpoint_{i}.csv')\n",
        "\n",
        "    time.sleep(DELAY_S)\n",
        "\n",
        "# ─── CONSTRUIR DATAFRAME FINAL ────────────────────────────────────────────────\n",
        "conv = pd.DataFrame(results)\n",
        "\n",
        "def calc_score_satisfaccion(row):\n",
        "    '''Score 0-100. CSAT declarado tiene prioridad; si no, usa score_riesgo_cx del LLM.'''\n",
        "    if pd.notna(row.get('csat_declarado')):\n",
        "        return int(round(((int(row['csat_declarado']) - 1) / 6.0) * 100))\n",
        "    base = int(np.clip((10 - row.get('score_riesgo_cx', 5)) * 10, 0, 100))\n",
        "    if row.get('acuerdo_pago', 0) == 1:\n",
        "        base = min(base + 10, 100)\n",
        "    if row.get('requiere_escalamiento', False):\n",
        "        base = max(base - 20, 0)\n",
        "    return base\n",
        "\n",
        "conv['score_satisfaccion'] = conv.apply(calc_score_satisfaccion, axis=1)\n",
        "\n",
        "print(f'\\n[OK] Pipeline LLM completado: {len(conv)} conversaciones procesadas')\n",
        "print(f'     Acuerdos detectados:      {conv[\"acuerdo_pago\"].sum()} ({conv[\"acuerdo_pago\"].mean()*100:.1f}%)')\n",
        "print(f'     Requieren escalamiento:   {conv[\"requiere_escalamiento\"].sum()}')\n",
        "print(f'     Score CX promedio:        {conv[\"score_satisfaccion\"].mean():.1f}/100')\n",
        "print()\n",
        "print('=== DISTRIBUCION DE TONOS DEL CLIENTE (CLASIFICACION LLM) ===')\n",
        "print(conv['tono_cliente'].value_counts().to_string())\n",
        "conv.head(3)"
    ]
}

# ── Índices a reemplazar en el notebook ──────────────────────────────────────
# Identificar las celdas por su id o contenido
cells = nb['cells']

# Encontrar índices de las celdas a reemplazar
idx_sec3_md = None   # markdown "## 3. Prompt Engineering"
idx_sec3_code = None  # code de MOTIVOS (heurísticas)
idx_sec4_md = None   # markdown "## 4. Agregación"
idx_sec4_code = None  # code de agregación y CSAT

for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    if '## 3.' in src and 'Prompt Engineering' in src and cell['cell_type'] == 'markdown':
        idx_sec3_md = i
    elif 'MOTIVOS' in src and 'disputa_saldo_o_cobro' in src and cell['cell_type'] == 'code':
        idx_sec3_code = i
    elif '## 4.' in src and 'Agregaci' in src and cell['cell_type'] == 'markdown':
        idx_sec4_md = i
    elif 'flatten_unique' in src and 'conv = df_raw' in src and cell['cell_type'] == 'code':
        idx_sec4_code = i

print(f'Indices encontrados:')
print(f'  sec3_md={idx_sec3_md}, sec3_code={idx_sec3_code}')
print(f'  sec4_md={idx_sec4_md}, sec4_code={idx_sec4_code}')

# Reemplazar en orden inverso para no perder indices
replacements = sorted([
    (idx_sec3_md, CELL_MD_SEC3),
    (idx_sec3_code, CELL_CODE_SEC3),
    (idx_sec4_md, CELL_MD_SEC4),
    (idx_sec4_code, CELL_CODE_SEC4),
], key=lambda x: x[0], reverse=True)

for idx, new_cell in replacements:
    if idx is not None:
        cells[idx] = new_cell
        print(f'  [OK] Celda {idx} reemplazada -> {new_cell["id"]}')
    else:
        print(f'  [WARN] Indice no encontrado para {new_cell["id"]}')

# Guardar notebook
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'\n[OK] Notebook guardado: {NB_PATH}')
print(f'     Total celdas: {len(nb["cells"])}')
