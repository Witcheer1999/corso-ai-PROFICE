from crewai.tools import tool
from pathlib import Path
from typing import List
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    def __init__(self):
        self.persist_dir = "faiss_index_medical"
        self.k = 4
        
        self.embeddings = self._get_embeddings()
        self.llm = self._get_llm()
        self.vector_store = None
        self.chain = None
        
        self._initialize_rag()
    
    def _get_embeddings(self):
        """Inizializza embedding Azure OpenAI"""
        api_key = os.getenv("AZURE_API_KEY")
        return AzureOpenAIEmbeddings(
            model="text-embedding-ada-002",
            azure_endpoint=os.getenv("AZURE_API_BASE"),
            api_key=api_key
        )
    
    def _get_llm(self):
        """Inizializza LLM Azure OpenAI"""
        api_key = os.getenv("AZURE_API_KEY")
        endpoint = os.getenv("AZURE_API_BASE")
        deployment = os.getenv("MODEL")

        return AzureChatOpenAI(
            deployment_name=deployment,
            openai_api_version="2024-02-15-preview",
            azure_endpoint=endpoint,
            openai_api_key=api_key,
            temperature=0.1
        )
    
    def _create_medical_documents(self) -> List[Document]:
        """Crea documenti medici di esempio hardcodati"""
        medical_documents = [
            # Malattie respiratorie
            Document(
                page_content="""L'asma è una malattia respiratoria cronica caratterizzata da infiammazione delle vie aeree.
                Sintomi principali:
                - Dispnea (difficoltà respiratoria)
                - Tosse persistente, specialmente notturna
                - Respiro sibilante
                - Senso di oppressione toracica
                
                Cause:
                - Allergeni (acari, pollini, pelo di animali)
                - Irritanti (fumo, inquinamento)
                - Infezioni respiratorie
                - Esercizio fisico intenso
                - Stress emotivo
                
                Trattamento:
                - Broncodilatatori a breve durata (salbutamolo)
                - Corticosteroidi inalatori per controllo a lungo termine
                - Antileucotrieni
                - Evitare i trigger
                - Piano d'azione personalizzato per gestire le crisi""",
                metadata={"categoria": "respiratorio", "malattia": "asma"}
            ),
            
            Document(
                page_content="""L'influenza è una malattia respiratoria contagiosa causata dai virus influenzali.
                
                Sintomi tipici:
                - Febbre alta improvvisa (38-40°C)
                - Dolori muscolari e articolari
                - Mal di testa intenso
                - Tosse secca
                - Mal di gola
                - Stanchezza estrema
                - Naso che cola o congestionato
                
                Prevenzione:
                - Vaccinazione annuale
                - Lavaggio frequente delle mani
                - Evitare contatti con persone malate
                - Coprire bocca e naso quando si starnutisce
                
                Trattamento:
                - Riposo a letto
                - Idratazione abbondante
                - Antipiretici (paracetamolo, ibuprofene)
                - Antivirali se iniziati entro 48 ore (oseltamivir)""",
                metadata={"categoria": "infettivo", "malattia": "influenza"}
            ),
            
            # Malattie metaboliche
            Document(
                page_content="""Il diabete mellito è una malattia cronica caratterizzata da alti livelli di glucosio nel sangue.
                
                Diabete di Tipo 1:
                - Esordio in età giovane
                - Distruzione autoimmune delle cellule beta pancreatiche
                - Richiede insulina esogena
                
                Diabete di Tipo 2:
                - Più comune negli adulti
                - Resistenza all'insulina
                - Spesso associato a obesità
                
                Sintomi comuni:
                - Poliuria (minzione frequente)
                - Polidipsia (sete eccessiva)
                - Perdita di peso inspiegabile
                - Visione offuscata
                - Stanchezza cronica
                - Guarigione lenta delle ferite
                
                Gestione:
                - Monitoraggio glicemico regolare
                - Dieta equilibrata e controllo carboidrati
                - Esercizio fisico regolare
                - Farmaci orali (metformina) o insulina
                - Controllo del peso
                - Gestione dello stress""",
                metadata={"categoria": "metabolico", "malattia": "diabete"}
            ),
            
            # Malattie cardiovascolari
            Document(
                page_content="""L'ipertensione arteriosa è una condizione caratterizzata da pressione sanguigna costantemente elevata.
                
                Classificazione:
                - Normale: <120/80 mmHg
                - Elevata: 120-129/<80 mmHg
                - Ipertensione stadio 1: 130-139/80-89 mmHg
                - Ipertensione stadio 2: ≥140/90 mmHg
                
                Fattori di rischio:
                - Età avanzata
                - Storia familiare
                - Obesità
                - Sedentarietà
                - Dieta ricca di sodio
                - Stress cronico
                - Fumo e alcol
                
                Complicanze:
                - Infarto miocardico
                - Ictus
                - Insufficienza renale
                - Retinopatia
                
                Trattamento:
                - Modifiche dello stile di vita (dieta DASH)
                - ACE-inibitori
                - Beta-bloccanti
                - Diuretici
                - Calcio-antagonisti
                - Monitoraggio regolare""",
                metadata={"categoria": "cardiovascolare", "malattia": "ipertensione"}
            ),
            
            # Malattie gastrointestinali
            Document(
                page_content="""La gastrite è l'infiammazione della mucosa gastrica che può essere acuta o cronica.
                
                Cause principali:
                - Infezione da Helicobacter pylori
                - Uso prolungato di FANS
                - Consumo eccessivo di alcol
                - Stress grave
                - Reflusso biliare
                
                Sintomi:
                - Dolore epigastrico
                - Nausea e vomito
                - Sensazione di pienezza dopo i pasti
                - Perdita di appetito
                - Bruciore di stomaco
                - Eruttazione frequente
                
                Diagnosi:
                - Endoscopia con biopsia
                - Test per H. pylori
                - Esami del sangue
                
                Trattamento:
                - Inibitori di pompa protonica
                - Antibiotici per H. pylori
                - Antiacidi
                - Modifiche dietetiche
                - Evitare irritanti gastrici""",
                metadata={"categoria": "gastrointestinale", "malattia": "gastrite"}
            ),
            
            # Malattie neurologiche
            Document(
                page_content="""L'emicrania è un disturbo neurologico caratterizzato da mal di testa ricorrenti e intensi.
                
                Caratteristiche:
                - Dolore pulsante unilaterale
                - Durata 4-72 ore
                - Intensità moderata-severa
                
                Sintomi associati:
                - Nausea e vomito
                - Fotofobia (sensibilità alla luce)
                - Fonofobia (sensibilità ai suoni)
                - Aura visiva nel 25% dei casi
                
                Trigger comuni:
                - Stress
                - Cambiamenti ormonali
                - Alcuni alimenti (cioccolato, formaggi stagionati)
                - Alterazioni del sonno
                - Cambiamenti meteorologici
                
                Trattamento:
                - Triptani per attacchi acuti
                - FANS
                - Beta-bloccanti per profilassi
                - Antiepilettici preventivi
                - Tecniche di rilassamento
                - Diario delle emicranie""",
                metadata={"categoria": "neurologico", "malattia": "emicrania"}
            ),
            
            # Allergie
            Document(
                page_content="""La rinite allergica è un'infiammazione della mucosa nasale causata da reazione allergica.
                
                Tipi:
                - Stagionale (febbre da fieno)
                - Perenne (tutto l'anno)
                
                Sintomi tipici:
                - Starnuti ripetuti
                - Rinorrea acquosa
                - Congestione nasale
                - Prurito nasale, oculare e palatale
                - Lacrimazione
                - Occhiaie allergiche
                
                Allergeni comuni:
                - Pollini (graminacee, alberi)
                - Acari della polvere
                - Pelo di animali domestici
                - Muffe
                
                Trattamento:
                - Antistaminici orali
                - Corticosteroidi nasali
                - Decongestionanti
                - Immunoterapia specifica
                - Evitare allergeni
                - Lavaggi nasali con soluzione salina""",
                metadata={"categoria": "allergico", "malattia": "rinite_allergica"}
            ),
            
            # Malattie infettive comuni
            Document(
                page_content="""La polmonite è un'infezione che infiamma gli alveoli polmonari.
                
                Agenti causali:
                - Batterici (Streptococcus pneumoniae più comune)
                - Virali
                - Fungini (in immunocompromessi)
                
                Sintomi:
                - Febbre alta con brividi
                - Tosse produttiva con espettorato
                - Dolore toracico pleuritico
                - Dispnea
                - Tachicardia
                - Confusione (negli anziani)
                
                Diagnosi:
                - Radiografia toracica
                - Esami del sangue (PCR, emocromo)
                - Coltura dell'espettorato
                
                Trattamento:
                - Antibiotici (amoxicillina, macrolidi)
                - Supporto respiratorio se necessario
                - Idratazione
                - Antipiretici
                - Fisioterapia respiratoria
                - Vaccinazione preventiva""",
                metadata={"categoria": "infettivo", "malattia": "polmonite"}
            )
        ]
        
        return medical_documents
    
    def _initialize_rag(self):
        """Inizializza o carica il sistema RAG"""
        if Path(self.persist_dir).exists():
            # Carica indice esistente
            self.vector_store = FAISS.load_local(
                self.persist_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Indice FAISS medico caricato da disco")
        else:
            # Crea nuovo indice
            documents = self._create_medical_documents()
            self.vector_store = FAISS.from_documents(
                documents,
                self.embeddings
            )
            # Salva su disco
            self.vector_store.save_local(self.persist_dir)
            print("✅ Nuovo indice FAISS medico creato e salvato")
        
        # Costruisci la chain RAG
        self._build_rag_chain()
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Formatta i documenti recuperati"""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)
    
    def _build_rag_chain(self):
        """Costruisce la chain RAG con LangChain"""
        # Template del prompt
        template = """Sei un assistente medico esperto. Usa le seguenti informazioni dal database medico per rispondere alla domanda.
        Se le informazioni non sono sufficienti, indicalo chiaramente.
        
        Contesto dal database:
        {context}
        
        Domanda: {question}
        
        Fornisci una risposta dettagliata, precisa e in italiano:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Crea il retriever
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.k}
        )
        
        # Costruisci la chain
        self.chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def search(self, question: str) -> str:
        """Esegue una ricerca RAG"""
        if not self.chain:
            return "❌ Sistema RAG non inizializzato correttamente"
        
        try:
            result = self.chain.invoke(question)
            return result
            
        except Exception as e:
            return f"❌ Errore nella ricerca RAG: {str(e)}"

# Istanza globale del sistema RAG
_rag_system = None

def get_rag_system():
    """Restituisce l'istanza singleton del sistema RAG"""
    global _rag_system
    if _rag_system is None:
        _rag_system = RAGSystem()
    return _rag_system

@tool
def search_rag(question: str) -> str:
    """
    Effettua una ricerca nel database medico locale utilizzando RAG.
    Restituisce informazioni mediche basate sui documenti hardcodati di esempio.
    """
    try:
        rag_system = get_rag_system()
        result = rag_system.search(question)
        return f"Risultato della ricerca medica per '{question}':\n\n{result}"
    except Exception as e:
        return f"Errore nella ricerca RAG: {str(e)}"