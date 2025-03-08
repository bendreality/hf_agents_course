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
    num_ctx=24576
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


def generate_specific_prompt(record: CompanyRecord) -> str:
    """
    Generiert einen präzisen, fokussierten Prompt basierend auf den verfügbaren Daten

    Args:
        record: Der zu analysierende Firmendatensatz

    Returns:
        str: Ein angepasster Prompt für den Agenten mit präzisen Suchstrategien
    """
    # Extrahiere präzise Bestandteile der Adresse für gezielte Suchen
    address_parts = record.address.split(',')[0].strip() if ',' in record.address else record.address.strip()

    base_prompt = f"""
    Du bist ein Experte für Firmenrecherche in Deutschland. Deine Aufgabe ist es, herauszufinden, 
    ob an der spezifischen Adresse "{address_parts}" in Leverkusen eine Firma existiert.

    WICHTIG: Konzentriere dich ausschließlich auf diese genaue Adresse. Sei präzise und effizient, 
    um das Context Window nicht unnötig zu belasten.

    Führe folgende präzise Suchanfragen durch:
    """

    # Baue präzise Suchanfragen basierend auf verfügbaren Daten
    search_queries = []

    # 1. Immer nach der konkreten Adresse suchen
    search_queries.append(f"Firmen an {address_parts}, Leverkusen")

    # 2. Wenn Firmenname bekannt, diesen mit der Adresse kombinieren
    if record.company:
        search_queries.append(f"{record.company} {address_parts} Leverkusen")

    # 3. Wenn Kontaktname bekannt, diesen gezielt mit der Adresse kombinieren
    if record.contact_name:
        search_queries.append(f"{record.contact_name} {address_parts} Leverkusen")

    # 4. Spezifische Suche nach der E-Mail-Domain, wenn vorhanden
    if record.email and '@' in record.email:
        email_domain = record.email.split('@')[1]
        search_queries.append(f"domain:{email_domain} {address_parts}")

    # Füge die konkreten Suchstrategien zum Prompt hinzu
    search_instructions = "\n".join([f"- Suche nach: \"{query}\"" for query in search_queries])
    base_prompt += f"\n{search_instructions}\n\n"

    # Anweisungen zur Informationsauswertung
    base_prompt += """
    Analyse:
    1. Prüfe, ob an der EXAKTEN Adresse eine Firma registriert ist
    2. Versuche den Firmennamen zu bestätigen, falls angegeben
    3. Verifiziere die Kontaktperson, falls angegeben
    4. Ignoriere alle Informationen, die nicht direkt mit dieser spezifischen Adresse zusammenhängen

    Hinweis: Für lokale LLMs ist es wichtig, nur relevante Informationen zu sammeln und
    irrelevante Details zu vermeiden, um das Context Window nicht zu überlasten.
    """

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
    - **Address Verification**: [Beschreibe präzise, was an dieser Adresse existiert]
    - **Company Verification**: [Bestätige/widerlege die Existenz der genannten Firma an dieser Adresse]
    - **Contact Person Verification**: [Bestätige/widerlege die Verbindung der Kontaktperson]

    ### Conclusion
    - **Company exists at this address**: [Yes/No/Uncertain]
    - **Confidence level**: [High/Medium/Low]
    - **Recommended action**: [Update database/Needs manual review/Delete record]

    ## Final Answer
    """

    return base_prompt + "\n\n" + data_summary + "\n\n" + output_format


def process_records(records: List[CompanyRecord], agent: CodeAgent) -> Dict[int, Dict]:
    """
    Verarbeitet einzelne Firmendatensätze mit präzisen Suchanfragen

    Args:
        records: Liste der zu verarbeitenden Firmendatensätze
        agent: Der CodeAgent für die Verarbeitung

    Returns:
        Dict: Ein Dictionary mit Ergebnissen für jeden Datensatz
    """
    results = {}

    # Verarbeite jeden Datensatz einzeln mit präzisen, fokussierten Anfragen
    for record in records:
        print(f"Verarbeite Datensatz ID {record.id}: {record.address}")

        # Erstelle einen präzisen Suchprompt für diesen spezifischen Datensatz
        prompt = generate_specific_prompt(record)

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

    # Erstelle Agenten mit optimierten Parametern für lokale LLMs
    agent = CodeAgent(
        model=model,
        tools=[
            # Reduziere die Anzahl der Ergebnisse, um das Context Window zu schonen
            CustomizableGoogleSearchTool(provider="serper", default_max_results=3),
            VisitWebpageTool(),
            write_to_markdown
        ],
        additional_authorized_imports=[],
        max_steps=6,  # Reduzierte Schritte für weniger Context-Verbrauch
        planning_interval=4
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

    # Verarbeite jeden Datensatz mit präzisen, fokussierten Anfragen
    results = process_records(valid_records, agent)

    # Erstelle einen Zusammenfassungsbericht
    summary_report = create_summary_report(results, valid_records)
    write_to_markdown(summary_report, "summary.md")
    print("Analyse abgeschlossen. Ergebnisse wurden in 'summary.md' gespeichert.")


if __name__ == "__main__":
    main()