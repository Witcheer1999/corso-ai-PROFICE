
---

# Project Work Finale 

**Titolo**: Costruzione di un Assistente AI Specializzato con CrewAI + RAG

## Obiettivo

Realizzare un **assistente AI specializzato in un dominio scelto** (es. turismo, sanità, educazione, legale, aziendale) che:

1. Usi **CrewAI** per orchestrare più agenti collaborativi.
2. Implementi un sistema **RAG (Retrieval-Augmented Generation)** per recuperare informazioni da una knowledge base specifica.
3. Produca **output utili e concreti** (es. report, consigli, piani di azione).
4. Includa un sistema di **valutazione** sia per il RAG (es. con Ragas) sia per gli agenti (con framework di analisi).
5. Includa la documentazione tecnica del progetto e un **documento di conformità all’EU AI Act**.

---

## Requisiti Minimi

### 1. CrewAI

* Almeno **due crew distinte** con ruoli diversi (es. *ResearchCrew*, *AnalysisCrew*, *SummaryCrew*).
* Una logica di flusso ben definita (inizio, step intermedi, output finale).
* Gestione dello stato e routing (anche semplice).

### 2. RAG

* Creare o selezionare una knowledge base (documenti PDF, articoli, dataset).
* Implementare la pipeline: **chunking → embedding → retrieval → risposta generata**.
* Integrare il RAG in uno degli agenti CrewAI.

### 3. Valutazione

* Usare **Ragas** per valutare il sistema RAG (metriche: relevance, faithfulness, answer correctness).
* Utilizzare un **framework di valutazione per CrewAI**:


### 4. Documentazione

* Redigere una **documentazione tecnica**:

  * Obiettivo del progetto
  * Architettura del sistema (schema dei moduli: CrewAI, RAG, valutazione)
  * Istruzioni di installazione ed esecuzione
  * Scelte progettuali e trade-off

* Scrivere un **documento EU AI Act**:

  * Livello di rischio del sistema (es. basso, limitato, alto → con motivazione)
  * Requisiti di trasparenza e gestione bias
  * Valutazione etica e possibili mitigazioni

---

## Funzionalità Extra (opzionali, dopo aver completato i requisiti minimi)

Una volta rispettati i punti fondamentali, i gruppi possono **estendere il progetto** con funzionalità aggiuntive a loro scelta, ad esempio:

* Interfaccia web (es. con Streamlit o Gradio).
* Implementazione di **agenti con personalità diverse** o strategie decisionali.
* Integrazione con **fonti esterne live** (API, web search, database).

---

## Consegna Finale

Ogni gruppo dovrà consegnare:

1. **Codice funzionante** (CrewAI + RAG + valutazione).
2. **README tecnico** con architettura e istruzioni.
3. **Documento EU AI Act**.
4. **Breve presentazione (15 min)** del progetto:

   * Descrizione obiettivo
   * Demo rapida
   * Risultati della valutazione

---

## Sistema di valutazione
1. **RAGAS E EVALUATION FRAMEWORK** 50%
2. **ARCHITETTURA E FUNZIONAMENTO CREWAI** 30%
3. **DOCUMENTAZIONE E EU AI ACT** 20%
4. **BONUSES** 10%



