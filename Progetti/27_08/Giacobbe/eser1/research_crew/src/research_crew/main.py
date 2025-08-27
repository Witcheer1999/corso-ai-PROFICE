from research_crew.crew import ResearchCrew

def run():
    inputs = {"topic": input("Inserisci l'argomento da ricercare: ")}

    result = ResearchCrew().crew().kickoff(inputs=inputs)

    # Print the result
    print("\n\n=== FINAL REPORT ===\n\n")
    print(result.raw)

    print("\n\nReport has been saved to output/risultato.md")

if __name__ == "__main__":
    run()