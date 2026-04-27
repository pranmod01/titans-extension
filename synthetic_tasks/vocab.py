"""Shared vocabulary, entity pools, and filler text generators."""

import random

# --- Entity pools ---

COUNTRIES = [
    "Valdoria", "Kesthen", "Morvaine", "Thalwick", "Surendia", "Bryndal",
    "Vexoria", "Plathenia", "Cromwall", "Delvara", "Norbeth", "Fyreholm",
    "Asturen", "Glenveth", "Prevonia", "Ulsmark", "Zarindel", "Hemdral",
    "Corvath", "Whitfen", "Dranmoor", "Saltbury", "Ironfeld", "Grenmoor",
    "Luventis", "Caldoria", "Thorngate", "Westmire", "Valkray", "Ostveld",
    "Brandholm", "Fenwick", "Dunmere", "Stormhall", "Greyvast", "Aldenmoor",
    "Northfen", "Rivermark", "Highveld", "Darkholm", "Crestfall", "Edenmoor",
    "Frostwick", "Goldmere", "Harrowfen", "Ironmoor", "Jadehollow", "Kinmere",
    "Lowveld", "Mistholm", "Nightfen", "Oakmere", "Pinemark", "Queensfen",
    "Ravenhollow", "Sandmoor", "Tidemark", "Underfen", "Valemoor", "Windholm",
    "Xalthor", "Yewmere", "Zephyrfen", "Ashveld", "Blazemark", "Copperfen",
    "Dawnhollow", "Embermark", "Flintmoor", "Galemark", "Hazelmere", "Ivyfen",
    "Jasperveld", "Kelpmoor", "Limemark", "Maplehollow", "Nettlemark", "Obsidianfen",
    "Pearlmoor", "Quartzmark", "Reedhollow", "Silverfen", "Thornmark", "Umbravel",
    "Violetmoor", "Willowfen", "Xylemmark", "Yellowhollow", "Zircmark", "Amberfen",
    "Basaltmoor", "Cedarmark", "Dolomiteven", "Ebonfen", "Feldsparmark", "Granitehollow",
    "Halitemark", "Igneousmoor", "Jettemark", "Kaolinfen", "Lateritemark", "Marblehollow",
]

CITIES = [
    "Alphaville", "Betaburg", "Gammacity", "Deltaport", "Epsilonhaven",
    "Zetaton", "Etatown", "Thetaville", "Iotaburg", "Kappafield",
    "Lambdacity", "Muport", "Nuhaven", "Xihaven", "Omicroncity",
    "Picity", "Rhotown", "Sigmaburg", "Tauville", "Upsilonport",
    "Phifield", "Chicity", "Psihaven", "Omegatown", "Northvale",
    "Southmark", "Eastholm", "Westmere", "Centralia", "Newbridge",
    "Oldport", "Highrock", "Lowfield", "Deepwater", "Clearview",
    "Brightholm", "Darkwood", "Coldspring", "Hotdale", "Dryfield",
    "Wetmoor", "Stonegate", "Ironbridge", "Copperton", "Silverton",
    "Goldfield", "Rubydale", "Emeraldport", "Sapphireton", "Diamondburg",
    "Pearlhaven", "Onyxville", "Garnettown", "Amethystport", "Topazfield",
    "Opalburg", "Jadeville", "Crystalport", "Quartzton", "Graniteburg",
    "Marbleton", "Slateville", "Limeston", "Sandstone", "Basaltburg",
    "Obsidianport", "Flintville", "Chalkton", "Claymoor", "Siltfield",
    "Loamburg", "Pebbleton", "Boulderport", "Cliffside", "Canyonburg",
    "Valleyville", "Ridgeton", "Plateauport", "Plainfield", "Marshburg",
    "Swampville", "Bogton", "Fenport", "Mirafield", "Dunhaven",
    "Moorburg", "Hillton", "Mountainport", "Forestville", "Meadowton",
    "Prairieport", "Desertburg", "Tundreville", "Taigatown", "Junglehaven",
    "Savannaton", "Steppeport", "Coastburg", "Bayhaven", "Inletville",
    "Fjordton", "Gulfport", "Capebury", "Peninsulaburg", "Islandhaven",
    "Archipeport", "Atollburg", "Reefville", "Shoalton", "Coveport",
    "Harborburg", "Portville", "Dockton", "Anchorhaven", "Marineburg",
    "Seafront", "Shoreline", "Coastmark", "Tideburg", "Waveton",
    "Surfhaven", "Breezeville", "Galeburg", "Stormport", "Thunderton",
    "Lightningburg", "Rainville", "Snowtown", "Frosthaven", "Iceport",
    "Glacierburg", "Blizzardville", "Hailton", "Sleetport", "Fogburg",
    "Mistville", "Cloudton", "Skyhaven", "Sunport", "Starburg",
    "Moonville", "Dawnton", "Duskhaven", "Twilightport", "Midnightburg",
    "Noonville", "Mornington", "Eveninghaven", "Springport", "Summerburg",
    "Autumnville", "Winterton", "Solsticeport", "Equinoxburg", "Seasonhaven",
    "Harveston", "Sowingport", "Bloomburg", "Fadeville", "Witherton",
    "Renewhaven", "Cycleport", "Tidalburg", "Lunarville", "Solaron",
    "Cosmosport", "Nebulaburg", "Galaxyville", "Starclusterton", "Voidhaven",
    "Etherport", "Aetherburg", "Plasmahaven", "Quantumville", "Atomton",
    "Moleculeport", "Cellburg", "Tissueville", "Organonton", "Systemhaven",
    "Spectrumport", "Prismburg", "Wavelengthville", "Frequencyton", "Amplitudehaven",
    "Resonanceport", "Harmonicburg", "Overtonevillle", "Undertonton", "Chordhaven",
    "Melodyport", "Rhythmburg", "Tempohaven", "Beatville", "Measureton",
    "Cadenceport", "Phraseburg", "Motifville", "Themeton", "Variationhaven",
    "Developmentport", "Recapburg", "Codaville", "Finelleton", "Epiloguehaven",
    "Prologueport", "Overtureburg", "Interludio", "Prestohaven", "Andanteton",
    "Allegroport", "Adagioburg", "Larghissimhaven", "Vivaceton", "Allegretto",
    "Moderatohaven", "Andantinoport", "Andantesburg", "Larghettohaven", "Largoton",
    "Gravehaven", "Lentissimport", "Sostenutoburg", "Lentohaven", "Ritardandon",
]

PERSON_NAMES = [
    "Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Liam", "Maria", "Nathan", "Olivia", "Paul",
    "Quinn", "Rachel", "Sam", "Tara", "Uma", "Victor", "Wendy", "Xavier",
    "Yara", "Zoe", "Aaron", "Beth", "Carl", "Diana", "Evan", "Fiona",
    "George", "Hannah", "Ivan", "Julia", "Kevin", "Laura", "Mike", "Nina",
    "Oscar", "Pam", "Quincy", "Rose", "Steve", "Tina", "Ulrich", "Vera",
    "Walter", "Xena", "Yolanda", "Zack", "Adrian", "Bella", "Chris", "Dora",
    "Ethan", "Flora", "Gabe", "Holly", "Igor", "Jenny", "Kent", "Lily",
    "Marco", "Nora", "Owen", "Petra", "Raj", "Sara", "Tom", "Ursula",
    "Vince", "Willa", "Xander", "Yasmin", "Zara", "Alec", "Bianca", "Cole",
    "Della", "Erik", "Faith", "Glen", "Hana", "Isak", "Jade", "Kurt",
    "Lena", "Milo", "Nell", "Omar", "Piper", "Ravi", "Sasha", "Troy",
    "Una", "Vale", "Wade", "Xia", "Yuki", "Zeno",
]

JOBS = [
    "teacher", "doctor", "engineer", "nurse", "pilot", "chef", "lawyer",
    "architect", "journalist", "accountant", "dentist", "pharmacist",
    "veterinarian", "librarian", "firefighter", "policeofficer", "paramedic",
    "electrician", "plumber", "carpenter", "painter", "mechanic", "welder",
    "farmer", "fisherman", "baker", "butcher", "florist", "gardener",
    "barber", "tailor", "shoemaker", "jeweler", "watchmaker", "optician",
    "photographer", "filmmaker", "musician", "dancer", "actor", "writer",
    "poet", "sculptor", "designer", "developer", "analyst",
    "biologist", "chemist", "physicist", "geologist", "astronomer",
]

PETS = [
    "dog", "cat", "rabbit", "hamster", "parrot", "goldfish", "turtle",
    "guineapig", "ferret", "lizard", "snake", "hedgehog", "chinchilla",
    "canary", "cockatiel", "iguana", "gecko", "chameleon", "tarantula",
    "scorpion", "crab", "lobster", "shrimp", "snail", "slug", "worm",
    "ant", "bee", "butterfly", "moth",
]

LEADERS = [
    "Aldric", "Breva", "Corwin", "Davan", "Elara", "Fendor", "Gala",
    "Harven", "Irene", "Jovan", "Kira", "Levan", "Mara", "Nevran", "Ophra",
    "Prex", "Queva", "Rolan", "Sova", "Trevak", "Ulan", "Voran",
    "Wela", "Xavra", "Yoran", "Zeva",
]

# Attributes usable as fact keys in task1
ATTRIBUTES = ["capital", "leader", "currency", "language", "anthem", "flag", "symbol"]

# City pool specifically for task1 values (answer tokens — single words only)
CITY_VALUES = [c for c in CITIES if " " not in c]

# --- Filler topic templates ---

FILLER_TOPICS = {
    "weather": [
        "The weather in {place} has been {adj} this {season} .",
        "Forecasters expect {adj} conditions across {place} this week .",
        "Heavy {precip} has delayed several flights out of {place} .",
        "Temperatures in {place} reached {extreme} levels yesterday .",
        "The {season} brought {adj} weather to the region of {place} .",
        "Officials in {place} warned residents of incoming {precip} .",
    ],
    "sports": [
        "The {place} team won their match by a score of {num} to {num2} .",
        "Athletes from {place} are preparing for the championship .",
        "The coach of the {place} squad announced a new training strategy .",
        "Fans in {place} celebrated after their team reached the finals .",
        "The tournament held in {place} attracted competitors from the region .",
        "A record crowd attended the game in {place} last evening .",
    ],
    "economy": [
        "The market in {place} showed {adj} growth last quarter .",
        "Trade in {place} has expanded over the past year .",
        "Several businesses in {place} reported increased demand this month .",
        "Investment in {place} has grown steadily over three years .",
        "The central bank of {place} adjusted interest rates recently .",
        "Consumer spending in {place} rose during the holiday period .",
    ],
    "travel": [
        "Visitors to {place} remark on the scenic views near the town center .",
        "The road between {place} and the coast was repaired last month .",
        "A new airline route connecting {place} to the capital was announced .",
        "The bridge near {place} was reopened after a lengthy inspection .",
        "{name} recently returned from a trip to {place} .",
        "Hotels in {place} reported high occupancy rates this season .",
    ],
    "science": [
        "Researchers published findings on climate patterns near {place} .",
        "A new species of plant was discovered in forests outside {place} .",
        "Scientists confirmed the soil in {place} is rich in rare minerals .",
        "The observatory near {place} recorded an unusual celestial event .",
        "A geological survey of {place} revealed ancient rock formations .",
        "A team from {place} completed a study on local water quality .",
    ],
    "work": [
        "{name} submitted the quarterly report ahead of schedule .",
        "The team meeting covered the new project timeline .",
        "{name} completed the assignment before the deadline .",
        "The office in {place} will undergo renovations next month .",
        "Several departments merged their operations to improve efficiency .",
        "{name} was recognized for outstanding contributions this quarter .",
    ],
}

_FILLER_ADJ = ["mild", "harsh", "warm", "cold", "dry", "humid", "stormy", "calm", "foggy", "clear"]
_FILLER_SEASONS = ["spring", "summer", "autumn", "winter"]
_FILLER_PRECIP = ["rain", "snow", "sleet", "hail"]
_FILLER_EXTREME = ["high", "low"]
_FILLER_NUMS = list(range(1, 10))


def _filler_sentence(rng, topic=None):
    if topic is None:
        topic = rng.choice(list(FILLER_TOPICS.keys()))
    template = rng.choice(FILLER_TOPICS[topic])
    return template.format(
        place=rng.choice(CITIES),
        name=rng.choice(PERSON_NAMES),
        adj=rng.choice(_FILLER_ADJ),
        season=rng.choice(_FILLER_SEASONS),
        precip=rng.choice(_FILLER_PRECIP),
        extreme=rng.choice(_FILLER_EXTREME),
        num=rng.choice(_FILLER_NUMS),
        num2=rng.choice(_FILLER_NUMS),
    )


def generate_filler(rng, target_chars, topic_mix=None):
    """Generate filler text of approximately target_chars characters."""
    sentences = []
    total = 0
    while total < target_chars:
        topic = rng.choice(topic_mix) if topic_mix else None
        s = _filler_sentence(rng, topic)
        sentences.append(s)
        total += len(s) + 1  # +1 for space separator
    return " ".join(sentences)