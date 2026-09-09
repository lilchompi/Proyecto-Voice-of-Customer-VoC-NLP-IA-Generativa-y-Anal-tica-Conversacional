# Proyecto Voice of Customer (VoC), NLP, IA Generativa & Analítica Conversacional

Solución integral de Inteligencia Artificial Generativa, Procesamiento de Lenguaje Natural (NLP) y Arquitectura RAG para la optimización de operaciones de cobranza bancaria por WhatsApp.

---

## 🎯 Objetivo de la Prueba Técnica
Analizar una **base suministrada y anonimizada/sintética de 42,607 interacciones distribuidas en 1,197 conversaciones** entre clientes y asesores de cobranza, con el fin de:
1. **Identificar los principales motivos de no pago** mediante análisis semántico con IA Generativa (evitando falsos positivos de regex).
2. **Medir la efectividad de ofertas y argumentos** bajo el criterio estricto de acuerdo de pago (**intención expresa + fecha específica de compromiso**).
3. **Resumir de forma ejecutiva el 100% de las conversaciones** (1,197 chats estructurados).
4. **Diagnosticar los casos con menor satisfacción** separando el **CSAT Observado (Encuesta 1 a 7)** del **Score de Satisfacción Estimado por IA (0 a 100)**.
5. **Diseñar e implementar un PoC funcional de Arquitectura RAG** con base de conocimiento curada, recuperación semántica vectorial y guardrails institucionales.

---

## 🚀 Cómo Ejecutar la Aplicación Streamlit

Para iniciar el Dashboard Ejecutivo interactivo y el PoC del Copiloto RAG en tu navegador:

```bash
# 1. Instalar dependencias requeridas
pip install -r requirements.txt

# 2. Iniciar la aplicación web
streamlit run app.py
```

La aplicación se abrirá automáticamente en `http://localhost:8501`.

### Módulos Interactivos de la Aplicación (`app.py`):
1. **📊 Dashboard Ejecutivo (EDA)**: Gráficos de alta definición respondiendo a las preguntas de negocio A (motivos), B (ofertas) y C (efectividad de argumentos).
2. **📑 Resumen de Conversaciones (Paginado)**: Explorador paginado de alto rendimiento con los 1,197 resúmenes ejecutivos e inspección de transcripciones completas.
3. **⚠️ Análisis CX - 5 Peores Calificadas**: Diagnóstico de detracción con selector de ranking dual:
   - *Ranking 1: CSAT Observado (Encuesta Real 1 a 7)*.
   - *Ranking 2: Score de Satisfacción Estimado por IA (0 a 100)*.
4. **🤖 PoC Real Copiloto RAG**: Motor funcional en tiempo real con Knowledge Base de 7 políticas, búsqueda vectorial con TF-IDF + Cosine Similarity, citas normativas, verificación de guardrails y evaluación automática de benchmark etiquetado.
5. **🏛️ Arquitectura RAG & Metodologías**: Especificación técnica de la solución RAG y 5 metodologías analíticas avanzadas (Uplift ML, BERTopic, Process Mining, Survival Analysis y Emotion Trajectory).

---

## 📓 Cómo Ejecutar el Notebook Jupyter

El notebook principal contiene el flujo analítico completo documentado paso a paso:

```bash
jupyter notebook Prueba_Tecnica_VoC_NLP_IA_Generativa.ipynb
```

### Estructura del Notebook:
- **Sección 1**: Contexto, objetivos de negocio y criterio estricto de acuerdo de pago.
- **Sección 2**: Ingesta y normalización de la base suministrada y anonimizada/sintética (42,607 turnos).
- **Sección 3**: Prompt Engineering y Esquema JSON Estructurado para IA Generativa.
- **Sección 4**: Pipeline Multi-Modelo resiliente (Ollama Cloud / Groq / Gemini) con guardado progresivo a prueba de bloqueos de Windows.
- **Sección 5**: Análisis Exploratorio de Datos (EDA) respondiendo preguntas A, B y C.
- **Sección 6**: Resumen ejecutivo del 100% de las conversaciones (Pregunta D).
- **Sección 7**: Análisis profundo de las 5 peores conversaciones (Pregunta E) con separación de CSAT Observado vs Estimado.
- **Sección 8**: Propuesta Arquitectónica RAG, Matriz Oferta-Motivo y Guardrails.
- **Sección 9**: Cinco Metodologías Analíticas Avanzadas propuestas y conclusiones de negocio.

---

## 📁 Estructura del Repositorio

```text
├── app.py                                   # Aplicación web Streamlit con EDA y PoC RAG funcional
├── Prueba_Tecnica_VoC_NLP_IA_Generativa.ipynb # Notebook Jupyter end-to-end documentado
├── propuesta_rag_estandarizacion.md         # Propuesta arquitectónica RAG y metodologías analíticas
├── presentacion_entregable.md               # Presentación ejecutiva y modelo Power BI (DAX / Star Schema)
├── requirements.txt                         # Dependencias del proyecto (pandas, streamlit, plotly, etc.)
├── .env                                     # Configuración de credenciales y modelos LLM
├── data/
│   └── Base sintetica conversaciones.xlsx   # Dataset de 42,607 interacciones / 1,197 conversaciones
└── outputs/
    ├── conversaciones_enriquecidas.csv      # 1,197 conversaciones enriquecidas con variables IA
    ├── motivos_no_pago.csv                  # Frecuencia y distribución de causas de mora
    ├── efectividad_ofertas.csv              # Ranking de efectividad (% Agreement Rate) por oferta
    ├── efectividad_argumentos.csv           # Ranking de efectividad (% Agreement Rate) por táctica
    ├── resumen_conversaciones.csv           # Resumen estructurado de las 1,197 conversaciones
    └── worst5_satisfaccion.csv              # Top peores casos (CSAT Observado y Estimado)
```

---

## 📊 Resumen de Resultados Principales

- **Tasa de Acuerdo (Agreement Rate)**: **31.41%** (376 de 1,197 conversaciones con compromiso de pago formal con fecha).
- **Causas Principales de No Pago**:
  - *Falta de liquidez*: **24.5%** (293 convs)
  - *Pago ya realizado*: **24.1%** (289 convs) — *Fricción por desfase contable*.
  - *Disputa de saldo*: **23.4%** (280 convs) — *Objeción a cuotas o seguros no contratados*.
- **Oferta más efectiva**: **Fraccionamiento en cuotas** con **47.1%** de acuerdos.
- **Argumento más efectivo**: **Evitar gastos adicionales** de cobranza con **48.3%** de acuerdos.
- **Satisfacción**: CSAT Observado promedio de **5.43 / 7** (en 235 encuestas reales) y Score Estimado de **52.8 / 100**.
