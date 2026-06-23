"""Cricket field presets stored as RH angle/radius positions."""

from Backends.src.analysis.field_geometry import fielder_from_position

PRESET_NAMES = [
    "Balanced",
    "Attacking Pace",
    "Defensive Pace",
    "Spin Attack",
    "T20 Death Overs",
    "Powerplay",
]

FIELD_PRESETS = {
    "Balanced": [
        ("Wicket Keeper", "Wicket Keeper"),
        ("First Slip", "First Slip"),
        ("Point", "Point"),
        ("Cover", "Cover"),
        ("Mid Off", "Mid Off"),
        ("Mid On", "Mid On"),
        ("Mid Wicket", "Mid Wicket"),
        ("Square Leg", "Square Leg"),
        ("Fine Leg", "Fine Leg"),
        ("Third Man", "Third Man"),
        ("Long Off", "Long Off"),
    ],
    "Attacking Pace": [
        ("Wicket Keeper", "Wicket Keeper"),
        ("First Slip", "First Slip"),
        ("Second Slip", "Second Slip"),
        ("Gully", "Gully"),
        ("Point", "Point"),
        ("Cover", "Cover"),
        ("Mid Off", "Mid Off"),
        ("Mid Wicket", "Mid Wicket"),
        ("Square Leg", "Square Leg"),
        ("Fine Leg", "Fine Leg"),
        ("Third Man", "Third Man"),
    ],
    "Defensive Pace": [
        ("Wicket Keeper", "Wicket Keeper"),
        ("Third Man", "Third Man"),
        ("Deep Point", "Deep Point"),
        ("Deep Cover", "Deep Cover"),
        ("Long Off", "Long Off"),
        ("Long On", "Long On"),
        ("Deep Mid Wicket", "Deep Mid Wicket"),
        ("Deep Square Leg", "Deep Square Leg"),
        ("Fine Leg", "Fine Leg"),
        ("Mid Off", "Mid Off"),
        ("First Slip", "First Slip"),
    ],
    "Spin Attack": [
        ("Wicket Keeper", "Wicket Keeper"),
        ("First Slip", "First Slip"),
        ("Short Fine Leg", "Short Fine Leg"),
        ("Square Leg", "Square Leg"),
        ("Mid Wicket", "Mid Wicket"),
        ("Cow Corner", "Cow Corner"),
        ("Mid On", "Mid On"),
        ("Mid Off", "Mid Off"),
        ("Cover", "Cover"),
        ("Point", "Point"),
        ("Long Off", "Long Off"),
    ],
    "T20 Death Overs": [
        ("Wicket Keeper", "Wicket Keeper"),
        ("Long Off", "Long Off"),
        ("Long On", "Long On"),
        ("Deep Cover", "Deep Cover"),
        ("Deep Mid Wicket", "Deep Mid Wicket"),
        ("Deep Square Leg", "Deep Square Leg"),
        ("Fine Leg", "Fine Leg"),
        ("Third Man", "Third Man"),
        ("Cow Corner", "Cow Corner"),
        ("Mid Off", "Mid Off"),
        ("Mid On", "Mid On"),
    ],
    "Powerplay": [
        ("Wicket Keeper", "Wicket Keeper"),
        ("First Slip", "First Slip"),
        ("Second Slip", "Second Slip"),
        ("Gully", "Gully"),
        ("Point", "Point"),
        ("Cover", "Cover"),
        ("Extra Cover", "Extra Cover"),
        ("Mid Off", "Mid Off"),
        ("Mid On", "Mid On"),
        ("Mid Wicket", "Mid Wicket"),
        ("Square Leg", "Square Leg"),
    ],
}


def create_preset_fielders(preset_name="Balanced"):
    layout = FIELD_PRESETS.get(preset_name, FIELD_PRESETS["Balanced"])
    return [fielder_from_position(name, position) for name, position in layout[:11]]
