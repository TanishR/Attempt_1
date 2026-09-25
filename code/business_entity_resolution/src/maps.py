# maps.py

NAME_ABBREVIATIONS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
    "intl": "international",
    "mfg": "manufacturing",
    "svcs": "services",
    "assoc": "associates",
    "bros": "brothers",
    "cie": "compagnie",
    "ste": "societe",
    "grp": "groupe",
    # additional french abbreviations
    "praivet": "private",
    "praibhet": "private",
    "piraivet": "private",
    "praivrr": "private",
    "pra": "private",
    "li": "limited",
    "limitet": "limited",
    "limirrd": "limited",
    "limtid": "limited",
    "elelpi": "llp"
}

LEGAL_SUFFIXES = {
    "private", "limited", "incorporated", "llc", "llp", "lp", "pc", "pllc",
    "corporation", "company", "sarl", "sas", "sasu", "sa", "sci", "eurl", "snc", 
    "public", "m/s", "cie", 
}

WEAK_TOKENS = {
    "(india)", "shri", "the", "usa", "france"
}

ADDR_ABBREVIATIONS = {
    "rd": "road",
    "street": "st",
    "st": "st",
    "saint": "st",
    "sainte": "ste",
    "ste": "ste",
    "suite": "ste",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "bd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "hwy": "highway",
    "apt": "apartment",
    "fl": "floor",
    "no": "number",
    "opp": "opposite",
    "nr": "near",
    "bldg": "building",
    "pl": "place",
    "chs": "society",
    # french words
    "all": "allee",
    "imp": "impasse",
    "ch": "chemin",
    "chem": "chemin",
    "rte": "route",
    "fg": "faubourg",
    "rue": "rue",
    "avenue": "avenue",
    "boulevard": "boulevard",
    "allee": "allee",
    "place": "place",
    "impasse": "impasse",
    "chemin": "chemin",
    "route": "route",
    "faubourg": "faubourg"
}

UNIT_TOKENS = {
    "po", "box", "pmb", "unit", "suite", "floor", "apartment", "apt", "fl"
}

STATE_MAP = {
    # US Full to 2-letter
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
    "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
    "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
    
    # India States (canonical and variants)
    "andaman and nicobar islands": "an", "andaman & nicobar": "an", "andaman": "an",
    "andhra pradesh": "ap", "andhra": "ap",
    "arunachal pradesh": "ar", "arunachal": "ar",
    "assam": "as",
    "bihar": "br",
    "chandigarh": "ch",
    "chhattisgarh": "cg", "chatishgarh": "cg", "chhatisgarh": "cg",
    "dadra and nagar haveli and daman and diu": "dn", "dadra and nagar haveli": "dn", "daman and diu": "dd",
    "delhi": "dl", "new delhi": "dl", "nct of delhi": "dl",
    "goa": "ga",
    "gujarat": "gj", "gujrat": "gj",
    "haryana": "hr",
    "himachal pradesh": "hp", "himachal": "hp",
    "jammu and kashmir": "jk", "jammu & kashmir": "jk", "j & k": "jk",
    "jharkhand": "jh",
    "karnataka": "ka", "karnatka": "ka",
    "kerala": "kl", "keralam": "kl",
    "ladakh": "la",
    "lakshadweep": "ld",
    "madhya pradesh": "mp",
    "maharashtra": "mh", "maharastra": "mh",
    "manipur": "mn",
    "meghalaya": "ml",
    "mizoram": "mz",
    "nagaland": "nl",
    "odisha": "or", "orissa": "or",
    "puducherry": "py", "pondicherry": "py",
    "punjab": "pb",
    "rajasthan": "rj", "rajastan": "rj",
    "sikkim": "sk",
    "tamil nadu": "tn", "tamilnadu": "tn",
    "telangana": "tg", "telengana": "tg", "ts": "tg",
    "tripura": "tr",
    "uttar pradesh": "up",
    "uttarakhand": "uk", "uttaranchal": "uk",
    "west bengal": "wb",

    # France Regions and Departments
    "nord": "hdf", "pas de calais": "hdf", "hauts de france": "hdf",
    "gironde": "naq", "nouvelle aquitaine": "naq",
    "loire atlantique": "pdl", "pays de la loire": "pdl",
    "ile de france": "idf", "paris": "idf",
    "bretagne": "bre",
    "normandie": "nor"
}

SHORT_STATE_MAP = {c: c for c in STATE_MAP.values()}
SHORT_STATE_MAP.update({
    "ts": "tg",
    "od": "or",
    "ua": "uk",
    "ct": "cg",
})
