RECOMMENDATIONS = {
    "hidden_fees": (
        "Present the complete applicable price, including mandatory "
        "fees and charges, before the user's decision point."
    ),

    "false_urgency": (
        "Use genuine availability information without artificial "
        "pressure, exaggerated scarcity, or misleading countdowns."
    ),

    "sneaking": (
        "Keep optional services unselected by default and require "
        "clear affirmative user consent."
    ),

    "misdirection": (
        "Use neutral wording and balanced visual emphasis so that "
        "meaningful alternatives are presented clearly."
    ),

    "interface_interference": (
        "Give meaningful choices comparable visual prominence and "
        "avoid steering users through disproportionate interface hierarchy."
    ),

    "obstruction": (
        "Provide a clear and accessible path for cancellation, "
        "modification, or other user-controlled actions."
    ),
}


def get_recommendation(pattern: str) -> str:
    return RECOMMENDATIONS.get(
        pattern,
        "Use clear, transparent, and user-respecting interface design."
    )