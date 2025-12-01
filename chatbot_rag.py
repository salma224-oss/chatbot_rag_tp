import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import torch
import numpy as np
import faiss
from typing import List, Dict, Tuple

# Configuration de la page
st.set_page_config(
    page_title="Système RAG - Question/Réponse",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre sobre
st.title("Système RAG - Question/Réponse Intelligent")
st.markdown("Système de retrieval et génération basé sur les documents NLP")

# Initialisation des états
if "history" not in st.session_state:
    st.session_state.history = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "system_ready" not in st.session_state:
    st.session_state.system_ready = False

# Sidebar minimaliste
with st.sidebar:
    st.header("Paramètres")
    
    st.subheader("Recherche")
    top_k = st.slider(
        "Documents à récupérer",
        min_value=1,
        max_value=5,
        value=3
    )
    
    st.subheader("Conversation")
    max_history = st.slider(
        "Historique conversationnel",
        min_value=1,
        max_value=5,
        value=3
    )
    
    max_tokens = st.slider(
        "Longueur des réponses",
        min_value=50,
        max_value=200,
        value=128
    )
    
    st.markdown("---")
    
    if st.button("Effacer l'historique", use_container_width=True):
        st.session_state.history = []
    
    if st.button("Redémarrer le système", use_container_width=True):
        st.session_state.rag_system = None
        st.session_state.system_ready = False
        st.session_state.history = []
        st.rerun()
    
    st.markdown("---")
    st.subheader("Statut")
    
    if st.session_state.system_ready:
        st.success("Système opérationnel")
    else:
        st.warning("Initialisation en cours")

# Classe RAG professionnelle
class RAGSystem:
    def __init__(self):
        self.embedder = None
        self.tokenizer = None
        self.gen_model = None
        self.documents = []
        self.index = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Chargement des modèles
            with st.spinner("Chargement du modèle d'embedding..."):
                self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            
            with st.spinner("Chargement du modèle génératif..."):
                self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
                self.gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
                self.gen_model = self.gen_model.to(self.device)
            
            # Base de connaissances
            self.documents = [
                "Le traitement automatique du langage naturel (NLP) est un domaine de l'intelligence artificielle qui vise à permettre aux machines de comprendre et générer du langage humain.",
                "Les modèles Transformers sont basés sur le mécanisme d'attention qui permet au modèle de se concentrer sur les parties importantes d'une séquence.",
                "Les embeddings transforment le texte en vecteurs denses capturant la sémantique. Des textes similaires ont des vecteurs proches dans l'espace d'embedding.",
                "Le fine-tuning adapte un modèle pré-entraîné à une tâche spécifique en réentraînant ses couches sur un dataset annoté.",
                "Le RAG combine la récupération de documents pertinents avec la génération de réponses pour fournir des informations factuelles.",
                "La recherche sémantique utilise des embeddings pour trouver des documents basés sur leur sens plutôt que des mots-clés exacts.",
                "Les applications NLP incluent les chatbots, la traduction automatique, l'analyse de sentiment et les systèmes de question-réponse."
            ]
            
            # Construction de l'index
            doc_embeddings = self.embedder.encode(
                self.documents,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            
            dim = doc_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(doc_embeddings)
            
            return True
            
        except Exception as e:
            st.error(f"Erreur d'initialisation: {str(e)}")
            return False
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None:
            return []
        
        try:
            query_embedding = self.embedder.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            scores, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.documents):
                    results.append({
                        "document_index": int(idx),
                        "score": float(score),
                        "content": self.documents[idx]
                    })
            
            return results
            
        except Exception as e:
            st.error(f"Erreur de recherche: {str(e)}")
            return []
    
    def generate_response(self, question: str, retrieved_docs: List[Dict], history: List, max_history: int = 3, max_tokens: int = 128) -> str:
        try:
            prompt = self._build_prompt(question, retrieved_docs, history, max_history)
            
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=1024
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.gen_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    temperature=0.7
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            return f"Erreur de génération: {str(e)}"
    
    def _build_prompt(self, question: str, retrieved_docs: List[Dict], history: List, max_history: int) -> str:
        history_text = ""
        for i, (q, a, _) in enumerate(history[-max_history:]):
            history_text += f"Question {i+1}: {q}\nRéponse: {a}\n\n"
        
        docs_text = "\n".join([
            f"Document {i+1} (score: {doc['score']:.3f}): {doc['content']}"
            for i, doc in enumerate(retrieved_docs)
        ])
        
        prompt = f"""Assistant expert en NLP.

Historique:
{history_text if history_text else "Nouvelle conversation."}

Documents de référence:
{docs_text}

Instructions:
- Répondre en français
- Utiliser uniquement les documents fournis
- Être précis et structuré
- Indiquer si l'information n'est pas disponible

Question: {question}

Réponse:"""
        
        return prompt

# Fonctions utilitaires
def initialize_system():
    if st.session_state.rag_system is None:
        rag_system = RAGSystem()
        if rag_system.initialize():
            st.session_state.rag_system = rag_system
            st.session_state.system_ready = True
            return True
    return st.session_state.system_ready

def get_rag_response(question: str, top_k: int = 3, max_history: int = 3, max_tokens: int = 128) -> Tuple[str, List[Dict]]:
    if not st.session_state.system_ready or st.session_state.rag_system is None:
        return "Système en cours d'initialisation", []
    
    rag_system = st.session_state.rag_system
    retrieved_docs = rag_system.semantic_search(question, top_k)
    answer = rag_system.generate_response(
        question, 
        retrieved_docs, 
        st.session_state.history,
        max_history,
        max_tokens
    )
    
    return answer, retrieved_docs

# Interface principale
def main():
    if not st.session_state.system_ready:
        st.header("Initialisation du système")
        with st.status("Chargement des composants...") as status:
            if initialize_system():
                status.update(label="Système initialisé", state="complete")
                st.success("Prêt pour utilisation")
            else:
                status.update(label="Erreur d'initialisation", state="error")
                return
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Interface de conversation")
        
        if st.session_state.history:
            st.subheader("Historique")
            for i, (question, answer, docs) in enumerate(st.session_state.history):
                with st.expander(f"Échange {i+1}: {question[:50]}...", expanded=(i == len(st.session_state.history)-1)):
                    st.write(f"**Question:** {question}")
                    st.write(f"**Réponse:** {answer}")
                    
                    with st.expander(f"Documents utilisés ({len(docs)})"):
                        for j, doc in enumerate(docs):
                            st.write(f"**Document {j+1}** (similarité: {doc['score']:.3f})")
                            st.write(doc["content"])
        else:
            st.info("Commencez par poser une question sur le NLP")
        
        # Chat interface
        user_question = st.chat_input("Votre question...")
        
        if user_question:
            with st.chat_message("user"):
                st.write(user_question)
            
            with st.chat_message("assistant"):
                with st.spinner("Recherche en cours..."):
                    answer, retrieved_docs = get_rag_response(
                        question=user_question,
                        top_k=top_k,
                        max_history=max_history,
                        max_tokens=max_tokens
                    )
                
                st.write(answer)
                
                if retrieved_docs:
                    avg_similarity = sum(doc['score'] for doc in retrieved_docs) / len(retrieved_docs)
                    st.caption(f"Similarité moyenne: {avg_similarity:.3f} - Documents: {len(retrieved_docs)}")
            
            st.session_state.history.append((user_question, answer, retrieved_docs))
            st.rerun()
    
    with col2:
        st.header("Questions de test")
        
        demo_questions = [
            "Qu'est-ce que le traitement automatique du langage naturel ?",
            "Comment fonctionnent les modèles Transformers ?",
            "Explique le concept d'embedding de phrase",
            "Qu'est-ce que l'architecture RAG ?",
            "Comment entraîne-t-on un modèle de classification ?"
        ]
        
        for question in demo_questions:
            if st.button(question, key=f"btn_{question}", use_container_width=True):
                answer, retrieved_docs = get_rag_response(
                    question=question,
                    top_k=top_k,
                    max_history=max_history,
                    max_tokens=max_tokens
                )
                st.session_state.history.append((question, answer, retrieved_docs))
                st.rerun()
        
        st.markdown("---")
        st.header("Architecture")
        
        st.markdown("""
        **Composants:**
        - Embeddings: all-MiniLM-L6-v2
        - Recherche: FAISS
        - Génération: Flan-T5
        - Base: 7 documents NLP
        """)

if __name__ == "__main__":
    main()