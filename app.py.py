"""
Application Streamlit - Géovisualisation Groupe
Analyse des départs de feu avant survenue d'un très grand feu entre 1973 et 2022
"""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import subprocess
import sys

# Configuration de la page
st.set_page_config(
    page_title="Analyse Incendies Prométhée",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Installation de gdown si nécessaire
try:
    import gdown
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'])
    import gdown

# Configuration
fichiers_drive = {
    "promethee_departements": "1tzMyoahsi3AJGmZMF9-d36QA33_awigh",
    "contour_promethee": "1vQg_Pi0j66Fx6bx9C-xZ3UJ3mDYYvft8",
    "communes_promethee": "186twOjMIpdm_-vtoj8Cxk9kS6_bJIWD-",
    "liste_incendies_du_20_09_2022_groupe": "1xY8DytJfgkSPGluXTeIF_Fh0Kkh0hQgK",
}

grands_feux = [
    {'annee': 1976, 'commune': 'Corbre-les-Cabanes', 'departement': '66', 'date': '1976-07-28', 'surface_ha': 6600},
    {'annee': 1979, 'commune': 'Le Luc', 'departement': '83', 'date': '1979-08-10', 'surface_ha': 5880},
    {'annee': 1990, 'commune': 'Collobrières', 'departement': '83', 'date': '1990-08-21', 'surface_ha': 9600},
    {'annee': 1990, 'commune': 'Vidauban', 'departement': '83', 'date': '1990-09-21', 'surface_ha': 11580},
    {'annee': 2003, 'commune': 'Vidauban', 'departement': '83', 'date': '2003-07-17', 'surface_ha': 6744},
    {'annee': 2003, 'commune': 'Vidauban', 'departement': '83', 'date': '2003-07-28', 'surface_ha': 5646},
    {'annee': 2003, 'commune': 'Santo-Pietro-di-Tenda', 'departement': '2B', 'date': '2003-08-29', 'surface_ha': 5532.51},
    {'annee': 2021, 'commune': 'Gonfaron', 'departement': '83', 'date': '2021-08-16', 'surface_ha': 6832}
]


@st.cache_data
def charger_donnees():
    """Charge toutes les données depuis Google Drive"""
    with st.spinner("Chargement des données depuis Google Drive..."):
        # Import du fichier incendies
        file_id_incendies_xlsx = fichiers_drive["liste_incendies_du_20_09_2022_groupe"]
        url_gdown_incendies_xlsx = f'https://drive.google.com/uc?id={file_id_incendies_xlsx}'
        output_temp_incendies_xlsx = 'liste_incendies_du_20_09_2022_groupe.xlsx'
        gdown.download(url_gdown_incendies_xlsx, output_temp_incendies_xlsx, quiet=True, fuzzy=True)
        
        df_incendies = pd.read_excel(output_temp_incendies_xlsx)
        
        # Import des GeoDataFrames
        file_id_dpt = fichiers_drive["promethee_departements"]
        url_gdown_dpt = f'https://drive.google.com/uc?id={file_id_dpt}'
        output_dpt = 'dpt_promethee.gpkg'
        gdown.download(url_gdown_dpt, output_dpt, quiet=True, fuzzy=True)
        promethee_dpt_shp = gpd.read_file(output_dpt)
        
        file_id_communes = fichiers_drive["communes_promethee"]
        url_gdown_communes = f'https://drive.google.com/uc?id={file_id_communes}'
        output_communes = 'communes_promethee.gpkg'
        gdown.download(url_gdown_communes, output_communes, quiet=True, fuzzy=True)
        communes_promethee = gpd.read_file(output_communes)
        
        return df_incendies, promethee_dpt_shp, communes_promethee


@st.cache_data
def preparer_donnees(df_incendies):
    """Prépare les données pour l'analyse"""
    col_date = 'Alerte'
    col_commune = 'Commune'
    col_dept = 'Département'
    col_surface = 'surf_ha'
    
    df_incendies['selected'] = 'non'
    
    if not pd.api.types.is_datetime64_any_dtype(df_incendies[col_date]):
        df_incendies[col_date] = pd.to_datetime(df_incendies[col_date])
    
    # Marquage des grands feux
    for feu in grands_feux:
        date_feu = pd.to_datetime(feu['date'])
        
        mask_date_dept = (
            (df_incendies[col_date].dt.date == date_feu.date()) &
            (df_incendies[col_dept].astype(str).str.strip() == str(feu['departement']))
        )
        
        candidats = df_incendies[mask_date_dept]
        
        if len(candidats) > 0:
            mask_finale = mask_date_dept & (
                df_incendies[col_commune].str.contains(feu['commune'], case=False, na=False)
            )
            
            if mask_finale.sum() > 0:
                df_incendies.loc[mask_finale, 'selected'] = 'oui'
            else:
                idx_max_surface = candidats[col_surface].idxmax()
                df_incendies.loc[idx_max_surface, 'selected'] = 'oui'
    
    return df_incendies, col_date, col_commune, col_dept, col_surface


@st.cache_data
def trouver_feux_temoins(_df_incendies, col_date, col_commune, col_dept, col_surface):
    """Trouve les feux témoins pour comparaison"""
    feux_temoins = []
    details_temoins = []
    
    for feu in grands_feux:
        date_feu = pd.to_datetime(feu['date'])
        commune_feu = feu['commune']
        annee_feu = feu['annee']
        mois_feu = date_feu.month
        
        mask_temoins = (
            (_df_incendies[col_surface] < 100) &
            (_df_incendies[col_commune].str.contains(commune_feu, case=False, na=False)) &
            (_df_incendies[col_date].dt.year != annee_feu)
        )
        
        candidats_temoins = _df_incendies[mask_temoins].copy()
        
        commune_origine = True
        if len(candidats_temoins) == 0:
            commune_origine = False
            mask_temoins_large = (
                (_df_incendies[col_surface] < 100) &
                (_df_incendies[col_date].dt.year != annee_feu)
            )
            candidats_temoins = _df_incendies[mask_temoins_large].copy()
            
            if len(candidats_temoins) > 0:
                candidats_temoins['mois'] = candidats_temoins[col_date].dt.month
                candidats_temoins['diff_mois'] = abs(candidats_temoins['mois'] - mois_feu)
                candidats_temoins = candidats_temoins.sort_values('diff_mois')
        
        if len(candidats_temoins) > 0:
            jour_annee_feu = date_feu.timetuple().tm_yday
            candidats_temoins['jour_annee'] = candidats_temoins[col_date].dt.dayofyear
            candidats_temoins['diff_jour_annee'] = abs(candidats_temoins['jour_annee'] - jour_annee_feu)
            candidats_temoins['diff_jour_annee'] = candidats_temoins['diff_jour_annee'].apply(
                lambda x: min(x, 365 - x)
            )
            candidats_temoins['diff_annee'] = abs(candidats_temoins[col_date].dt.year - annee_feu)
            candidats_temoins = candidats_temoins.sort_values(['diff_jour_annee', 'diff_annee'])
            
            feu_temoin = candidats_temoins.iloc[0]
            
            feux_temoins.append({
                'annee': feu_temoin[col_date].year,
                'commune': feu_temoin[col_commune],
                'departement': str(feu_temoin[col_dept]),
                'date': feu_temoin[col_date].strftime('%Y-%m-%d'),
                'surface_ha': feu_temoin[col_surface]
            })
            
            details_temoins.append({
                'Grand feu': f"{commune_feu} ({feu['date']})",
                'Témoin': f"{feu_temoin[col_commune]} ({feu_temoin[col_date].strftime('%Y-%m-%d')})",
                'Surface témoin (ha)': f"{feu_temoin[col_surface]:.1f}",
                'Même commune': "Oui" if commune_origine else "Non",
                'Écart (jours)': int(feu_temoin['diff_jour_annee']),
                'Écart (années)': int(feu_temoin['diff_annee'])
            })
        else:
            feux_temoins.append(None)
            details_temoins.append({
                'Grand feu': f"{commune_feu} ({feu['date']})",
                'Témoin': "Non trouvé",
                'Surface témoin (ha)': "-",
                'Même commune': "-",
                'Écart (jours)': "-",
                'Écart (années)': "-"
            })
    
    return feux_temoins, pd.DataFrame(details_temoins)


def calculer_departs(_df_incendies, col_date, periode_jours, feux_temoins):
    """Calcule le nombre de départs pour chaque période"""
    resultats = []
    
    for i, feu in enumerate(grands_feux):
        date_feu = pd.to_datetime(feu['date'])
        date_debut = date_feu - timedelta(days=periode_jours)
        
        mask_grand = (
            (_df_incendies[col_date] >= date_debut) &
            (_df_incendies[col_date] < date_feu)
        )
        nb_departs_grand = mask_grand.sum()
        
        resultats.append({
            'N°': i + 1,
            'Commune': feu['commune'],
            'Date': feu['date'],
            'Type': 'Grand feu',
            'Départs': nb_departs_grand
        })
        
        if feux_temoins[i] is not None:
            feu_temoin = feux_temoins[i]
            date_temoin = pd.to_datetime(feu_temoin['date'])
            date_debut_temoin = date_temoin - timedelta(days=periode_jours)
            
            mask_temoin = (
                (_df_incendies[col_date] >= date_debut_temoin) &
                (_df_incendies[col_date] < date_temoin)
            )
            nb_departs_temoin = mask_temoin.sum()
            
            resultats.append({
                'N°': i + 1,
                'Commune': feu_temoin['commune'],
                'Date': feu_temoin['date'],
                'Type': 'Feu témoin',
                'Départs': nb_departs_temoin
            })
    
    return pd.DataFrame(resultats)


def creer_graphique(df_graph, periode_jours):
    """Crée le graphique comparatif"""
    heights_grands = []
    heights_temoins = []
    labels = []
    
    for i in range(len(grands_feux)):
        subset = df_graph[df_graph['N°'] == i + 1]
        grand_feu_data = subset[subset['Type'] == 'Grand feu']
        temoin_data = subset[subset['Type'] == 'Feu témoin']
        
        if len(grand_feu_data) > 0:
            heights_grands.append(grand_feu_data.iloc[0]['Départs'])
            labels.append(f"{i+1}. {grands_feux[i]['commune']}\n{grands_feux[i]['date']}")
        
        if len(temoin_data) > 0:
            heights_temoins.append(temoin_data.iloc[0]['Départs'])
        else:
            heights_temoins.append(0)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(grands_feux))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, heights_grands, width, label='Grand feu (>1000ha)', color='#FF6B35')
    bars2 = ax.bar(x + width/2, heights_temoins, width, label='Feu témoin (<100ha)', color='#4ECDC4')
    
    ax.set_xlabel('Incendies comparés', fontsize=12, fontweight='bold')
    ax.set_ylabel('Nombre de départs de feux', fontsize=12, fontweight='bold')
    
    titre_periode = {30: "Mois précédent", 7: "Semaine précédente", 3: "3 jours précédents", 1: "Jour précédent"}
    ax.set_title(f'Comparaison du nombre de départs - {titre_periode.get(periode_jours, f"{periode_jours} jours")}\nGrand feu vs Feu témoin individuel', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig


# Interface Streamlit
def main():
    st.title("🔥 Analyse des Incendies - Zone Prométhée (1973-2022)")
    st.markdown("### Nombre de départs de feu avant survenue d'un très grand feu")
    st.markdown("*Script Cécile Larchey - M2 GMS*")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sélection de la période
        periode_option = st.selectbox(
            "Période d'analyse",
            ["Mois précédent (30 jours)", "Semaine précédente (7 jours)", "3 jours précédents", "Jour précédent"],
            index=0
        )
        
        periodes_map = {
            "Mois précédent (30 jours)": 30,
            "Semaine précédente (7 jours)": 7,
            "3 jours précédents": 3,
            "Jour précédent": 1
        }
        periode_jours = periodes_map[periode_option]
        
        st.markdown("---")
        st.markdown("### 📊 À propos")
        st.info("""
        Cette application analyse les départs de feux précédant 8 grands incendies historiques 
        et les compare avec des feux témoins (<100ha) pour détecter des patterns.
        """)
        
        st.markdown("### 🔥 Grands feux analysés")
        st.markdown(f"**{len(grands_feux)} incendies** de 1976 à 2021")
    
    # Chargement des données
    try:
        df_incendies, promethee_dpt_shp, communes_promethee = charger_donnees()
        st.success(f"✓ Données chargées : {len(df_incendies):,} incendies")
        
        # Préparation
        df_incendies, col_date, col_commune, col_dept, col_surface = preparer_donnees(df_incendies)
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Graphique", "📋 Tableaux", "🎯 Feux témoins", "ℹ️ Grands feux"])
        
        with tab1:
            st.subheader(f"Comparaison pour {periode_option}")
            
            # Calcul
            feux_temoins, df_details_temoins = trouver_feux_temoins(df_incendies, col_date, col_commune, col_dept, col_surface)
            df_graph = calculer_departs(df_incendies, col_date, periode_jours, feux_temoins)
            
            # Graphique
            fig = creer_graphique(df_graph, periode_jours)
            st.pyplot(fig)
            
            # Statistiques
            col1, col2, col3 = st.columns(3)
            grands_departs = df_graph[df_graph['Type'] == 'Grand feu']['Départs'].values
            temoins_departs = df_graph[df_graph['Type'] == 'Feu témoin']['Départs'].values
            
            with col1:
                st.metric("Départs moyens (Grands feux)", f"{np.mean(grands_departs):.1f}")
            with col2:
                st.metric("Départs moyens (Feux témoins)", f"{np.mean(temoins_departs):.1f}")
            with col3:
                diff = np.mean(grands_departs) - np.mean(temoins_departs)
                st.metric("Différence", f"{diff:.1f}", delta=f"{diff:.1f}")
        
        with tab2:
            st.subheader("Tableau récapitulatif")
            
            # Formater le tableau
            df_display = df_graph.copy()
            df_display = df_display[['N°', 'Commune', 'Date', 'Type', 'Départs']]
            
            # Coloration
            def colorize_row(row):
                if row['Type'] == 'Grand feu':
                    return ['background-color: #FFE5DD'] * len(row)
                else:
                    return ['background-color: #D4F1F4'] * len(row)
            
            st.dataframe(
                df_display.style.apply(colorize_row, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            # Téléchargement
            csv = df_display.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 Télécharger le tableau (CSV)",
                data=csv,
                file_name=f"analyse_incendies_{periode_jours}jours.csv",
                mime="text/csv"
            )
        
        with tab3:
            st.subheader("Feux témoins sélectionnés")
            st.markdown("*Feux de référence (<100ha) utilisés pour la comparaison*")
            
            st.dataframe(df_details_temoins, use_container_width=True, hide_index=True)
            
            st.info("""
            **Critères de sélection** :
            - Surface < 100 ha
            - Même commune (ou commune proche si non disponible)
            - Même période de l'année (jour de l'année proche)
            - Année différente du grand feu
            """)
        
        with tab4:
            st.subheader("Les 8 grands feux analysés")
            
            df_grands_feux = pd.DataFrame(grands_feux)
            df_grands_feux = df_grands_feux[['annee', 'commune', 'departement', 'date', 'surface_ha']]
            df_grands_feux.columns = ['Année', 'Commune', 'Département', 'Date', 'Surface (ha)']
            
            st.dataframe(df_grands_feux, use_container_width=True, hide_index=True)
            
            # Carte récapitulative
            st.markdown("### 📍 Répartition géographique")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Département 83 (Var)", "5 incendies")
                st.metric("Département 2B (Haute-Corse)", "1 incendie")
            with col2:
                st.metric("Département 66 (Pyrénées-Orientales)", "1 incendie")
                st.metric("Surface totale brûlée", f"{sum([f['surface_ha'] for f in grands_feux]):,.0f} ha")
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
