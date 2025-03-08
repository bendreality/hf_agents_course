import math
import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Any
import os
from smolagents import CodeAgent, GoogleSearchTool, VisitWebpageTool, LiteLLMModel, tool

# Vereinfachte Modellkonfiguration - nur ein Modell behalten
model = LiteLLMModel(
    model_id="ollama/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    api_key="noone needs an api key",
    num_ctx=8192
)


# Verbesserte Datenstruktur für Firmendatensätze
@dataclass
class CompanyRecord:
    id: int
    contact_name: str = ""
    company: str = ""
    address: str = ""
    email: str = ""

    def is_valid(self) -> bool:
        """Prüft, ob der Datensatz die Mindestanforderungen erfüllt"""
        return self.id > 0 and len(self.address) > 0

    def to_dict(self) -> Dict[str, Any]:
        """Konvertiert den Datensatz in ein Dictionary"""
        return {
            "ID": self.id,
            "Name of contact": self.contact_name,
            "Company": self.company,
            "Address": self.address,
            "Email": self.email
        }


# Google-Suchtool mit anpassbarer Ergebnisanzahl
class CustomizableGoogleSearchTool(GoogleSearchTool):
    name = "customizable_web_search"
    description = """Performs a google web search for your query then returns a string of the top search results.
    Allows customizing the number of results to display."""
    inputs = {
        "query": {"type": "string", "description": "The search query to perform."},
        "filter_year": {
            "type": "integer",
            "description": "Optionally restrict results to a certain year",
            "nullable": True,
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of search results to return",
            "default": 4,
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, provider: str = "serpapi", default_max_results: int = 4):
        super().__init__(provider=provider)
        self.default_max_results = default_max_results

    def forward(self, query: str, filter_year: Optional[int] = None, max_results: Optional[int] = None) -> str:
        # Verwende den Standardwert, wenn max_results nicht angegeben ist
        if max_results is None:
            max_results = self.default_max_results

        # Stelle sicher, dass max_results eine positive ganze Zahl ist
        if max_results <= 0:
            max_results = self.default_max_results

        # Rufe die übergeordnete Methode auf, um die Suchergebnisse zu erhalten
        all_results = super().forward(query, filter_year)

        # Wenn keine Ergebnisse zurückgegeben wurden, gib die Nachricht unverändert zurück
        if all_results.startswith("No results found"):
            return all_results

        # Trenne den Header von den tatsächlichen Ergebnissen
        header = "## Search Results\n"
        results_list = all_results.replace(header, "").split("\n\n")

        # Begrenze die Anzahl der Ergebnisse
        limited_results = results_list[:max_results]

        # Setze die Ausgabe wieder zusammen
        return header + "\n\n".join(limited_results)


@tool
def write_to_markdown(text: str, filename: str) -> bool:
    """
    Schreibt Text in eine Markdown-Datei

    Args:
        text: Der Text, der in die Markdown-Datei geschrieben werden soll
        filename: Der Name der Datei

    Returns:
        bool: True, wenn erfolgreich geschrieben wurde, sonst False
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"Text wurde erfolgreich in '{filename}' geschrieben.")
        return True
    except Exception as e:
        print(f"Fehler beim Schreiben der Datei: {e}")
        return False


def extract_postal_code(address: str) -> str:
    """Extrahiert die Postleitzahl aus einer Adresse"""
    # Suche nach typischem deutschen PLZ-Muster (5 Ziffern)
    match = re.search(r'\b(\d{5})\b', address)
    if match:
        return match.group(1)
    return "00000"  # Fallback, wenn keine PLZ gefunden wurde


def generate_prompt(record: CompanyRecord) -> str:
    """
    Generiert einen angepassten Prompt basierend auf den verfügbaren Daten

    Args:
        record: Der zu analysierende Firmendatensatz

    Returns:
        str: Ein angepasster Prompt für den Agenten
    """
    base_prompt = """
    Du bist ein Experte für Firmenrecherche in Deutschland. Deine Aufgabe ist es, zu überprüfen, 
    ob eine Firma an der angegebenen Adresse existiert.

    Nutze die zur Verfügung stehenden Tools, um Websuchen durchzuführen und Webseiten zu besuchen.
    Sei gründlich aber effizient in deiner Recherche.

    Suchstrategie:
    1. Überprüfe die Adresse und suche nach Firmen an diesem Standort
    2. Wenn ein Firmenname vorhanden ist, suche gezielt nach dieser Firma
    3. Wenn ein Kontaktname vorhanden ist, suche nach Verbindungen zu diesem Namen
    4. Berücksichtige, dass alle Unternehmen in Leverkusen, Deutschland liegen
    """

    # Anpassung des Prompts basierend auf verfügbaren Daten
    if record.company:
        base_prompt += f"\n\nAchte besonders auf den Firmennamen: {record.company}"
    if record.contact_name:
        base_prompt += f"\n\nSuche nach Firmen, die mit '{record.contact_name}' verbunden sind"
    if record.email:
        base_prompt += f"\n\nPrüfe, ob du die E-Mail-Adresse '{record.email}' bestätigen kannst"

    # Hinzufügen der zu analysierenden Daten
    data_summary = f"""
    ID: {record.id}
    Person/Kontakt: {record.contact_name}
    Firma: {record.company}
    Adresse: {record.address}
    E-Mail: {record.email}
    """

    output_format = """
    Output Format:

    ## Analysis of Company Existence

    ### Provided Information
    - **ID**: 
    - **Name of contact**: 
    - **Company**: 
    - **Address**: 
    - **Email**: 

    ### Analysis
    - **Verification on Google Maps**:
    - **Search for Companies Located at the Specific Address**: 
    - **Business Registration Information**: 

    ### Conclusion
    - **Company exists**: [Yes/No/Uncertain]
    - **Confidence level**: [High/Medium/Low]
    - **Recommended action**: [Update database/Needs manual review/Delete record]

    ## Final Answer
    """

    return base_prompt + "\n\n" + data_summary + "\n\n" + output_format


def process_records_by_location(records: List[CompanyRecord], agent: CodeAgent) -> Dict[int, Dict]:
    """
    Verarbeitet Datensätze gruppiert nach Standort für effizientere Anfragen

    Args:
        records: Liste der zu verarbeitenden Firmendatensätze
        agent: Der CodeAgent für die Verarbeitung

    Returns:
        Dict: Ein Dictionary mit Ergebnissen für jeden Datensatz
    """
    # Gruppiere nach PLZ
    records_by_location = {}
    for record in records:
        postal_code = extract_postal_code(record.address)
        if postal_code not in records_by_location:
            records_by_location[postal_code] = []
        records_by_location[postal_code].append(record)

    # Verarbeite Gruppen sequentiell
    results = {}
    for postal_code, location_records in records_by_location.items():
        print(f"Verarbeite Standort PLZ {postal_code} mit {len(location_records)} Datensätzen...")

        # Führe einmal eine allgemeine Recherche zu dieser Location durch, wenn mehr als ein Datensatz
        location_context = ""
        if len(location_records) > 1:
            location_query = f"Welche Unternehmen befinden sich im Postleitzahlenbereich {postal_code} in Leverkusen, Deutschland?"
            try:
                location_context = agent.run(location_query)
                print(f"Allgemeine Standortinformationen für PLZ {postal_code} abgerufen.")
            except Exception as e:
                print(f"Fehler bei der Standortrecherche: {e}")

        # Verarbeite dann einzelne Records
        for record in location_records:
            prompt = generate_prompt(record)
            if location_context:
                prompt += f"\n\nAllgemeine Informationen zum Standort:\n{location_context}"

            try:
                analysis = agent.run(prompt)
                write_to_markdown(analysis, f"{record.id}.md")
                results[record.id] = evaluate_results(record.id, analysis)
                print(f"Datensatz {record.id} erfolgreich verarbeitet.")
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Datensatz {record.id}: {e}")
                results[record.id] = {
                    "record_id": record.id,
                    "error": str(e),
                    "confidence": 0,
                    "needs_manual_review": True
                }

    return results


def evaluate_results(record_id: int, analysis: str) -> Dict:
    """
    Wertet die Analyseergebnisse aus und berechnet ein Konfidenzlevel

    Args:
        record_id: Die ID des analysierten Datensatzes
        analysis: Die Analyseergebnisse als Text

    Returns:
        Dict: Ein Dictionary mit Auswertungsergebnissen
    """
    # Extrahiere Schlüsselinformationen aus den Ergebnissen
    confidence_score = 0
    exists = "uncertain"

    # Überprüfe auf Schlüsselwörter und -phrasen
    analysis_lower = analysis.lower()

    # Existenzhinweise
    if "company exists: yes" in analysis_lower:
        exists = "yes"
        confidence_score += 5
    elif "company exists: no" in analysis_lower:
        exists = "no"
        confidence_score += 3

    # Vertrauenswürdigkeit der Quellen
    if "multiple sources confirm" in analysis_lower:
        confidence_score += 3
    if "verified on google maps" in analysis_lower or "found on google maps" in analysis_lower:
        confidence_score += 2
    if "official website found" in analysis_lower:
        confidence_score += 3
    if "business registration confirmed" in analysis_lower:
        confidence_score += 4

    # Qualität der Informationen
    if "contradicting information" in analysis_lower:
        confidence_score -= 2
    if "outdated information" in analysis_lower:
        confidence_score -= 1
    if "no information found" in analysis_lower:
        confidence_score -= 3

    # Normalisiere den Score
    if confidence_score < 0:
        confidence_score = 0
    elif confidence_score > 10:
        confidence_score = 10

    # Bestimme das Konfidenzlevel
    if confidence_score >= 7:
        confidence_level = "high"
    elif confidence_score >= 4:
        confidence_level = "medium"
    else:
        confidence_level = "low"

    # Empfehlung
    if exists == "yes" and confidence_level in ["high", "medium"]:
        recommendation = "update database"
    elif exists == "no" and confidence_level == "high":
        recommendation = "delete record"
    else:
        recommendation = "needs manual review"

    return {
        "record_id": record_id,
        "exists": exists,
        "confidence_score": confidence_score,
        "confidence_level": confidence_level,
        "recommendation": recommendation,
        "needs_manual_review": confidence_level == "low" or recommendation == "needs manual review"
    }


def create_summary_report(results: Dict[int, Dict], records: List[CompanyRecord]) -> str:
    """
    Erstellt einen Zusammenfassungsbericht der Analyseergebnisse

    Args:
        results: Die Auswertungsergebnisse für jeden Datensatz
        records: Die analysierten Firmendatensätze

    Returns:
        str: Ein Markdown-formatierter Bericht
    """
    report = "# Analyse-Zusammenfassung\n\n"

    # Statistiken
    total = len(results)
    exists = sum(1 for r in results.values() if r.get("exists") == "yes")
    not_exists = sum(1 for r in results.values() if r.get("exists") == "no")
    uncertain = total - exists - not_exists
    needs_review = sum(1 for r in results.values() if r.get("needs_manual_review", True))

    report += f"## Statistik\n\n"
    report += f"- **Gesamt analysiert**: {total}\n"
    report += f"- **Existierende Unternehmen**: {exists}\n"
    report += f"- **Nicht existierende Unternehmen**: {not_exists}\n"
    report += f"- **Ungewiss**: {uncertain}\n"
    report += f"- **Benötigen manuelle Überprüfung**: {needs_review}\n\n"

    # Detaillierte Ergebnisse
    report += "## Detaillierte Ergebnisse\n\n"

    # Tabellenkopf
    report += "| ID | Kontakt | Firma | Existiert | Konfidenz | Empfehlung |\n"
    report += "|---|---|---|---|---|---|\n"

    # Tabelleninhalt
    for record in records:
        result = results.get(record.id, {})
        exists_status = result.get("exists", "error")
        confidence = result.get("confidence_level", "low")
        recommendation = result.get("recommendation", "needs manual review")

        report += f"| {record.id} | {record.contact_name} | {record.company} | {exists_status} | {confidence} | {recommendation} |\n"

    return report


def main():
    """Hauptfunktion zur Steuerung des Programmablaufs"""
    # Beispieldaten aus der ursprünglichen Datei
    data_list = [
        [4, "Herr Lupo", "", "Hitdorfer Str. 205, 51371 Leverkusen", ""],
        [5, "", "", "Bergische Landstr. 67, 51375 Leverkusen, Schlebusch", ""],
        [6, "", "Creditreform", "Hitdorfer Str. 205, 51371 Leverkusen", ""],
        [7, "", "Creditreform 02/2002", "Bergische Landstr. 78, 51375 Leverkusen", ""],
        [8, "", "Kfz Aufbereitung", "Höhenstr. 38, 51381 Leverkusen", "kuehnmi5@gmail.com"],
        [9, "ulrich dost", "Bayer 04 Leverkusen Fußball GmbH", "122-124, 51373 Leverkusen", ""]
    ]

    # Erstelle Agenten
    agent = CodeAgent(
        model=model,
        tools=[
            CustomizableGoogleSearchTool(provider="serper", default_max_results=6),
            VisitWebpageTool(),
            write_to_markdown
        ],
        additional_authorized_imports=[],
        max_steps=20,
        planning_interval=4  # Beibehalten des Planning-Intervalls
    )

    # Konvertiere Daten in CompanyRecord-Objekte
    records = [CompanyRecord(
        id=item[0],
        contact_name=item[1] if len(item) > 1 else "",
        company=item[2] if len(item) > 2 else "",
        address=item[3] if len(item) > 3 else "",
        email=item[4] if len(item) > 4 else ""
    ) for item in data_list]

    # Filtere ungültige Datensätze
    valid_records = [r for r in records if r.is_valid()]
    print(f"Verarbeite {len(valid_records)} von {len(records)} gültigen Datensätzen...")

    # Verarbeite Datensätze gruppiert nach Standort
    results = process_records_by_location(valid_records, agent)

    # Erstelle einen Zusammenfassungsbericht
    summary_report = create_summary_report(results, valid_records)
    write_to_markdown(summary_report, "summary.md")
    print("Analyse abgeschlossen. Ergebnisse wurden in 'summary.md' gespeichert.")


if __name__ == "__main__":
    main()