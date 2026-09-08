# Proyecto Voice of Customer (VoC), NLP, IA Generativa & Analítica Conversacional

Este proyecto desarrolla una solución integral de inteligencia artificial y analítica conversacional para el análisis de interacciones de cobranza por WhatsApp entre asesores y clientes de una entidad financiera.

---

## 🚀 Cómo Ejecutar la Aplicación Streamlit

Para iniciar el Dashboard Ejecutivo interactivo y el Simulador Copiloto RAG en tu navegador:

```bash
# 1. Activar el entorno virtual (si aplica)
.\.venv\Scripts\activate

# 2. Iniciar la aplicación Streamlit
streamlit run app.py
```

La aplicación web se abrirá automáticamente en `http://localhost:8501`.

---

## 📊 Estructura de la Aplicación Streamlit (`app.py`)

1. **📊 Dashboard Ejecutivos (EDA)**: Gráficos interactivos de Plotly respondiendo a las preguntas de negocio A (motivos de no pago), B (ofertas del asesor) y C (efectividad/agreement rate).
2. **📑 Resumen de Conversaciones**: Tabla interactiva y tarjetas desplegables con el resumen estructurado en 4 viñetas de cada una de las 25 llamadas/chats.
3. **⚠️ Análisis CX - 5 Peores Calificadas**: Diagnóstico del Top 5 de conversaciones con menor satisfacción (CSAT), identificando factores determinantes y recomendaciones de mejora.
4. **🤖 Simulador Copiloto RAG**: Chatbot interactivo de prueba para asesores que recibe la objeción del cliente y genera la respuesta estandarizada con citas a políticas y guardrails de cumplimiento.
5. **🏛️ Arquitectura RAG & Metodologías**: Explicación técnica del pipeline RAG (Semantic Chunking, Hybrid Search Vector+BM25, Reranker, Guardrails) y 5 metodologías analíticas avanzadas (Uplift ML, BERTopic, Process Mining Conversacional, Survival Analysis y Emotion Trajectory Tracking).

---

## 📁 Archivos del Proyecto

- `app.py`: Aplicación web interactiva en Streamlit.
- `Prueba_Tecnica_VoC_NLP_IA_Generativa.ipynb`: Notebook Jupyter ejecutable documentado con prompts, NLP, EDA y gráficos.
- `propuesta_rag_estandarizacion.md`: Propuesta arquitectónica RAG completa y metodologías analíticas avanzadas.
- `presentacion_entregable.md`: Presentación ejecutiva lista para alta dirección y guía de integración en Power BI (Medidas DAX y Star Schema).
- `scripts/generate_dataset.py`: Script generador de dataset sintético de 25 conversaciones de cobranza.
- `scripts/analyze_voc.py`: Pipeline de enriquecimiento de variables, cálculo cuantitativo de CSAT y exportación de CSVs/PNGs.
- `data/conversaciones_whatsapp.csv`: Dataset principal de 100 turnos conversacionales.
- `outputs/`: Archivos CSV exportados y gráficos de alta resolución para Power BI.

---

## ⚙️ Requisitos e Instalación

```bash
pip install -r requirements.txt
pip install streamlit plotly
```
