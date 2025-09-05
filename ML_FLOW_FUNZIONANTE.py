from pydantic import BaseModel
from crewai.flow.flow import Flow, start, listen
from guide_creator_flow.crews.search_crew.search_crew import SearchCrew
import os
import time
from datetime import datetime

import mlflow  # MLflow tracking & evaluation
import pandas as pd  # <-- necessario per mlflow.evaluate con dataframe

from dotenv import load_dotenv
load_dotenv()

# ---------- MLflow base config ----------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
mlflow.autolog()  # autolog per openai/langchain/crewai dove supportato
mlflow.set_experiment("WebSearchFlowExperiment")


class WebSearchState(BaseModel):
    """State container for the WebSearch flow."""
    user_query: str = ""
    summary: str = ""
    answer: str = ""
    context: str = ""          # se vuoi passare contesto ai judge
    ground_truth: str = ""     # opzionale: se vuoi attivare similarity/correctness


class WebSearchFlow(Flow[WebSearchState]):
    """Interactive web search flow orchestrated via crewAI."""

    @start()
    def get_user_input(self):
        print("\n🔍 WEB SEARCH FLOW")
        query = input("📝 Inserisci il tuo argomento di ricerca: ").strip()
        if not query:
            query = "Ultime innovazioni nell'intelligenza artificiale 2024"
            print(f"➡️ Query di esempio: {query}")
        self.state.user_query = query

        # Log query di input
        mlflow.log_param("user_query", query)
        mlflow.log_metric("user_query_length_chars", len(query))
        mlflow.log_metric("user_query_length_words", len(query.split()))
        return query

    @listen(get_user_input)
    def search_and_summarize(self, query: str):
        print("\n🌐 Eseguo la ricerca con DuckDuckGo...")
        started = time.perf_counter()

        crew = SearchCrew().crew()
        result = crew.kickoff(inputs={"query": query})

        duration = time.perf_counter() - started
        self.state.summary = str(result)

        # Metriche/artefatti ricerca
        mlflow.log_metric("search_duration_seconds", duration)
        mlflow.log_metric("search_results_chars", len(self.state.summary))
        mlflow.log_metric("search_results_words", len(self.state.summary.split()))
        mlflow.log_metric("search_results_lines", self.state.summary.count("\n") + 1 if self.state.summary else 0)
        mlflow.log_text(self.state.summary, "search_summary.txt")
        return self.state.summary

    @listen(search_and_summarize)
    def display_results(self, summary: str):
        print("\n📋 RISULTATI DELLA RICERCA WEB\n")
        print(f"🔍 Query: {self.state.user_query}\n")
        print(summary)

        # --- LLM-as-a-judge con MLflow ---
        try:
            eval_metrics = self._run_llm_judge_mlflow(
                user_query=self.state.user_query,
                prediction=summary,
                context=self.state.context or None,              # opzionale
                ground_truth=self.state.ground_truth or None,    # opzionale
            )
            # Salvo snapshot metriche anche come dict (facile da leggere)
            if eval_metrics:
                mlflow.log_dict(eval_metrics, "eval_metrics_snapshot.json")
                mlflow.set_tag("llm_judge_status", "success")
        except Exception as e:
            mlflow.set_tag("llm_judge_status", f"failed:{type(e).__name__}")
            mlflow.log_text(str(e), "llm_judge_error.txt")

        return "Flow completato!"

    # ---------- Nuovo: judge con mlflow.evaluate ----------
    def _run_llm_judge_mlflow(
        self,
        user_query: str,
        prediction: str,
        context: str | None = None,
        ground_truth: str | None = None,
    ):
        """
        Usa i judge integrati MLflow:
          - answer_relevance (richiede inputs+predictions)
          - faithfulness (se fornisci context)
          - answer_similarity/answer_correctness (se fornisci ground_truth)
          - toxicity (metric non-LLM)
        Le metriche e la tabella vengono loggate automaticamente nel run attivo.
        """
        # Tabella di valutazione a 1 riga (scalabile a molte righe)
        data = {
            "inputs": [user_query],
            "predictions": [prediction],
        }
        if context is not None:
            data["context"] = [context]
        if ground_truth is not None:
            data["ground_truth"] = [ground_truth]

        df = pd.DataFrame(data)

        # Costruisci lista metriche in base alle colonne disponibili
        extra_metrics = [
            mlflow.metrics.genai.answer_relevance(),  # sempre se hai inputs+predictions
            mlflow.metrics.toxicity(),                # metrica non-LLM (HF pipeline)
        ]
        if "context" in df.columns:
            extra_metrics.append(mlflow.metrics.genai.faithfulness(context_column="context"))
        if "ground_truth" in df.columns:
            extra_metrics.extend([
                mlflow.metrics.genai.answer_similarity(),
                mlflow.metrics.genai.answer_correctness(),
            ])

        # model_type:
        # - "text" va bene per generico testo
        # - "question-answering" se passi ground_truth in stile QA
        model_type = "question-answering" if "ground_truth" in df.columns else "text"

        results = mlflow.evaluate(
            data=df,
            predictions="predictions",
            targets="ground_truth" if "ground_truth" in df.columns else None,
            model_type=model_type,
            extra_metrics=extra_metrics,
            evaluators="default",
        )
        # MLflow ha già loggato metriche e tabella 'eval_results_table'
        return results.metrics


def kickoff() -> None:
    """Entrypoint usato dal comando `crewai run` o `python -m guide_creator_flow.main`."""
    run_started_wall_time = datetime.utcnow().isoformat() + "Z"
    run_timer = time.perf_counter()
    with mlflow.start_run():
        mlflow.set_tag("app_name", "guide_creator_flow")
        mlflow.set_tag("flow_name", "WebSearchFlow")
        mlflow.set_tag("run_started_at_utc", run_started_wall_time)
        mlflow.set_tag("environment", os.getenv("APP_ENV", "local"))

        flow = WebSearchFlow()
        flow.kickoff()

        mlflow.log_metric("run_duration_seconds", time.perf_counter() - run_timer)


def plot() -> None:
    flow = WebSearchFlow()
    plot_method = getattr(flow, "plot", None)
    if callable(plot_method):
        plot_method()
    else:
        print("Plot is not supported by this Flow implementation.")


if __name__ == "__main__":
    kickoff()





# NOTE: aggiungere nel .env per far funzionare LLM as a judge di MLFLOW
# OPENAI_API_TYPE=azure
# OPENAI_API_KEY=---
# OPENAI_API_BASE=---
# OPENAI_API_VERSION=---
# OPENAI_DEPLOYMENT_NAME="gpt-4o"   # or your deployment name