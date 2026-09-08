# Presentación Ejecutiva - Proyecto Voice of Customer (VoC), NLP, IA Generativa & Analítica Conversacional

## 📌 Ficha Técnica del Proyecto (Clasificación IA Granular)
- **Origen de Datos**: `Base sintetica conversaciones.xlsx` (42,607 mensajes de WhatsApp / 1,197 conversaciones únicas).
- **Entorno de Procesamiento**: Python 3.14, Pandas, Openpyxl, Scikit-Learn, Seaborn/Matplotlib, Streamlit, Plotly, JSON Schema LLM Prompts, RAG Architecture.
- **Métricas de la Cartera Real**: Tasa de acuerdo de pago estricta: **7.0%** | Score Promedio de Satisfacción: **61.8/100**.

---

## 1. Preguntas de Negocio y Resultados Encontrados (EDA Granular sobre 1,197 Conversaciones)

### 📊 Pregunta A: Principales Causas y Motivos de No Pago (`motivos_no_pago`)
El motor de IA Generativa y NLP Granular analizó la conversación completa extrayendo causas exclusivamente de mensajes emitidos por el cliente (evitando sesgos por plantillas HSM o palabras genéricas del bot):

1. **Desconexión o Abandono del Chat (52.7% / 631 conv.)**: Sesiones expiradas o clientes que no respondieron tras espera de asignación de asesor.
2. **Falta de Liquidez (25.0% / 299 conv.)**: Clientes que expresan atraso en pago de nómina, iliquidez o imposibilidad de cubrir la cuota actual.
3. **Pago Ya Realizado (6.7% / 80 conv.)**: Clientes que reportan haber pagado mediante PSE o abono previo y reclaman por notificaciones indebidas.
4. **Priorización de Otros Gastos (6.6% / 79 conv.)**: Destinación prioritaria de ingresos a vivienda, salud, educación o alimentación.
5. **Desempleo (5.8% / 69 conv.)**: Pérdida de empleo o ingresos independientes.
6. **Problemas de Salud (2.4% / 29 conv.)**: Incapacidades médicas, cirugías o compra de medicamentos.
7. **Disputa de Saldo / Reclamos de Cobro (2.0% / 24 conv.)**: Inconformidad con el valor de la cuota, incremento no explicado o cobros a no titulares.
8. **Consultas o Trámites Administrativos (1.7% / 20 conv.)**: Solicitud de extractos o certificados.
9. **Emergencia Familiar (0.2% / 2 conv.)**: Calamidades domésticas expresas y corroboradas.

> **Visualización**: El gráfico generado sobre los datos reales está disponible en `outputs/chart_motivos_no_pago.png`.

---

## 2. Análisis Profundo de las 5 Conversaciones con Menor Satisfacción (Top 5 Peores CSAT)

Se analizaron en detalle los 5 chats con menor satisfacción declarada (CSAT 0/1 - Totalmente Insatisfecho, NPS 0, CES 5 - Muy Difícil):

| Conv ID | Motivo Real / Objeción | Fricción Principal (Fases 2 y 3) | CSAT Declarado | Factores Negativos Identificados | Recomendación Concreta de Mejora |
| :---: | :---: | :--- | :---: | :--- | :--- |
| **CONV_00000018** | Pago Ya Realizado | Exigencia de filtro de seguridad estricto cuando el cliente no tiene datos a la mano y solicita desvincular cobros. | **0.0 / 7** | Cobranza a cliente al día; gestor niega ayuda o bloqueo de mensajes. | Integración en tiempo real con pasarelas de pago PSE para pausar notificaciones automáticas. |
| **CONV_00000042** | Acuerdo Cancelado sin Notificar | Cliente esperó 15 días propuesta de acuerdo y el banco la canceló sin avisar. Gestor cierra chat para "no duplicar información". | **0.0 / 7** | Cancación silenciosa de trámite; evasión del gestor cerrando la sesión. | Control en CRM que impida al gestor cerrar el chat mientras exista trámite de reestructuración activo. |
| **CONV_00000062** | Descuento Nómina No Reflejado | Cliente al día por nómina es rebotado repetidamente entre Servicio al Cliente y WhatsApp de cobranzas. | **0.0 / 7** | Rebote infinito entre canales sin visibilidad de convenios de nómina. | Módulo omnicanal en CRM de cobranza para validar descuentos por nómina en tiempo real. |
| **CONV_00000083** | Reestructuración Impagable ($4M) | Cuota de reestructuración salta a $4M al final. HSM promete asesoría pero gestor solo exige abono sin modificar el plan. | **0.0 / 7** | Promesa HSM engañosa; cuota balón impagable en la reestructuración. | Eliminar cuotas balón al final de negociaciones y alinear HSM con atribuciones reales del gestor. |
| **CONV_00000089** | Cobranza a Línea Corporativa | WhatsApp de Constructora no deudora recibe llamadas y cobros. Gestor exige cédula a la recepcionista. | **0.0 / 7** | Cobranza a no titular; insistencia robótica sin validar la propiedad del número. | Protocolo de desvinculación de líneas corporativas tras 2 intentos fallidos de validación. |

---

## 3. Esquema JSON Estructurado & Flujo Conversacional en 4 Fases

### 🤖 1. Schema JSON Mode para LLMs
```json
{
  "chat_id": "CONV_00000018",
  "motivo_no_pago": "Pago Ya Realizado / Desvinculación de Mensajes",
  "submotivo_no_pago": "Cobranza repetitiva a cliente al día sin flexibilidad de validación",
  "oferta_asesor": "Exigencia de datos / Negativa a consultar o desvincular",
  "acuerdo_pago": false,
  "fecha_compromiso_pago": null,
  "csat_score": 1,
  "nps_score": 0,
  "ces_score": 5,
  "factores_negativos_csat": [
    "Cobranza a cliente que ya pagó",
    "Negativa del gestor a revisar pagos o bloquear mensajes",
    "Falta de flexibilidad en validación de identidad"
  ],
  "resumen_ejecutivo": "El cliente manifiesta estar al día y solicita no recibir más cobros. El gestor niega la consulta sin validación de datos previa, causando molestia extrema y abandono con CSAT 1."
}
```

### 🔬 2. Flujo Conversacional en 4 Fases Operativas
- **Fase 1: Autenticación e Intención Inicial**: Contacto voluntario o respuesta a campaña HSM.
- **Fase 2: Objeción del Cliente y Punto de Fricción**: Reclamos por cobros no reconocidos, cuotas balón o pagos de nómina no cargados.
- **Fase 3: Ruptura de Experiencia y Rebote de Canales**: Respuestas rígidas del gestor o remisión cruzada a líneas telefónicas.
- **Fase 4: Evaluación y Detracción Registrada**: Confirmación de estado detractor crítico en la encuesta post-chat.

---

## 4. Propuesta Arquitectónica RAG para Estandarización de Respuestas

- **Knowledge Base**: Manuales de políticas de cobranza, catálogo de ofertas por tramo de mora, guiones legales.
- **Pipeline Vectorial**: Semantic Chunking (300 tokens) + Dense Vectors (`text-embedding-3-small`) + Qdrant Vector DB.
- **Motor de Búsqueda Híbrida**: Dense Retrieval + BM25 Sparse Search + Cross-Encoder Reranker (`bge-reranker-large`).
- **Guardrails**: Reglas de cumplimiento que bloquean amenazas y promesas fuera de norma.
- **Criterios de Evaluación**: Faithfulness $>98\%$, Compliance $100\%$, Hallucination Rate $<0.01\%$, Latencia $<1.2\text{s}$, Post-RAG Agreement Uplift $+20\%$.

---

## 5. Recomendaciones Estratégicas de Negocio

1. **Reconciliación de Pagos en Tiempo Real (PSE / Nómina)**: Pausar enrutamiento automático de cobranza si existe reporte de pago en 24h.
2. **Alertas de Trámites Vigentes en CRM**: Prohibir el cierre unilateral de chats cuando existan solicitudes de reestructuración pendientes.
3. **Desvinculación Asistida de Líneas Corporativas**: Permitir liberar números que corresponden a empresas no deudoras sin exigir la cédula del titular.
4. **Despliegue del Copiloto RAG** en el frontend del gestor para sugerencias de negociación flexibles y estandarizadas.
