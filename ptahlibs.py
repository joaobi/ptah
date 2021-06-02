# Dictionary for the Unicode values for each sign Gardiner sign
gardiner_dict ={
    # 1 consonants
    'AA1': {'hex': 0x1340d, 'dec':'8861'},
    'D21': {'hex': 0x1308b, 'dec':'7963'},
    'D36': {'hex': 0x1309d, 'dec':''},
    'D46' : {'hex' : 0x130a7, 'dec':7991},
    'D58': {'hex': 0x130c0, 'dec':''},
    'G1': {'hex': 0x1313f, 'dec':''},
    'M17': {'hex': 0x131cb, 'dec':''},
    'G43': {'hex': 0x13171, 'dec':''},
    'Q3': {'hex': 0x132aa, 'dec':''},
    'I9': {'hex': 0x13191, 'dec':''},
    'G17': {'hex': 0x13153, 'dec':''},
    'N35': {'hex': 0x13216, 'dec':''},
    'O4': {'hex': 0x13254, 'dec':'8420'},
    'V28': {'hex': 0x1339b, 'dec':'8747'},
    'F32' : {'hex' : 0x13121, 'dec':8113},
    'S29' : {'hex' : 0x132f4, 'dec':8580},
    'N37' : {'hex' : 0x13219, 'dec':8361},
    'N29' : {'hex' : 0x1320e, 'dec':8350},
    'V31' : {'hex' : 0x133a1, 'dec':8753},
    'W11' : {'hex' : 0x133bc, 'dec':8780},
    'X1' : {'hex' : 0x133cf, 'dec':8799},
    'V13' : {'hex' : 0x1337f, 'dec':8719},
    'I10' : {'hex' : 0x13193, 'dec':8227},
    
    # 2 consonants / biliteral sings
    'F31' : {'hex' : 0x1311f, 'dec':''}, #ms
    'Y5' : {'hex' : 0x133e0, 'dec':''}, #mn
    'R11' : {'hex' : 0x132bd, 'dec':''}, #Djed
    'V30' : {'hex' : 0x1339f, 'dec':''}, #nb (Neb)
    'X8' : {'hex' : 0x133d9, 'dec':''}, #dj (Di)  
    'D37' : {'hex' : 0x1309e, 'dec':''}, #dj (Di)  
    'D28' : {'hex' : 0x13093, 'dec':''}, #ka  
    'D35' : {'hex' : 0x1309C, 'dec':''}, #nj (Ni)
    # '' : {'hex' : 0x, 'dec':''}, #XX
    

    # 3 consonants / triliteral signs
    'L1' : {'hex' : 0x131a3, 'dec':''}, #hper (kheper)    
    'S34' : {'hex' : 0x132f9, 'dec':''}, #anh (Ankh)
    'R8' : {'hex' : 0x132b9, 'dec':''}, #ntr (Netjer)
    'F35' : {'hex' : 0x13124, 'dec':''}, #nfr (Nefer)  


    # A	Man and his occupations
    'A1' : {'hex' : 0x13000, 'dec':''}, #Seated man (I)
    'A21' : {'hex' : 0x13019, 'dec':''}, # man holding staff with handkerchief: Civil Servant (sr), Courtier (smr), High Official, strike (achwj)
    'A40' : {'hex' : 0x1302D, 'dec':''}, #seated god, Ptah, Month (mnṯw) divine/heavenly I (j),(jnk), me, myself (wj) God
    'A30' : {'hex' : 0x13022, 'dec':''},

    # G - Birds
    'G26' : {'hex' : 0x1315D, 'dec':''}, # sacred Ibis on standard Ibis (hb) Id. ḏḥwtj	God Thoth, the god of scribes
    'G36' : {'hex' : 0x13168, 'dec':''},
    'G36A' : {'hex' : 0x13169, 'dec':''},
    'G38' : {'hex' : 0x1316C, 'dec':''},
    'G39' : {'hex' : 0x1316D, 'dec':''},        

    # H	Parts of birds
    'H6' : {'hex' : 0x13184, 'dec':''}, 

    # M	Trees and plants
    'M40' : {'hex' : 0x131E9, 'dec':''}, # bundle of reeds
    'M5' : {'hex' : 0x131B4, 'dec':''}, 

    # N -  Sky, earth, water
    'N16' : {'hex' : 0x131FE, 'dec':''}, # Land with grains
    'N17' : {'hex' : 0x131FF, 'dec':''}, # Land

    # S	Crowns, dress, staves, etc.
    'S40' : {'hex' : 0x13300, 'dec':''},

    # T	Warfare, hunting, butchery
    'T34' : {'hex' : 0x13330, 'dec':''}, # butcher's knife
    'T35' : {'hex' : 0x13331, 'dec':''}, # butcher's knife

    # U	Agriculture, crafts, and professions
    'U36' : {'hex' : 0x1335B, 'dec':''}, # fuller's-club

    # Y	- Writings, games, music
    'Y1' : {'hex' : 0x133DB, 'dec':''},
    'Y3' : {'hex' : 0x133DE, 'dec':''}, #scribe's outfit (mnhd) abbreviation for write, (ssch) writing,
    
    #Other - Z (strokes, signs derived from hieratic, geometrical features)
    'Z1' : {'hex' : 0x133e4, 'dec':''}, # single (vertical) stroke
    'Z4' : {'hex' : 0x133ED, 'dec':''}, # Dual stroke. Egyptian numeral 2, plural, majority, collective concept (e.g. meat, jwf), duality
}

# Fonts used for dataset generation 
hiero_fonts = {
    "AaronUMdCAlpha100" : {'ext':".ttf",'font_size':190,'adjust':0},
    "JSeshFont" : {'ext':".ttf",'font_size':140,'adjust':0},
    "Aegyptus" : {'ext':".otf",'font_size':190,'adjust':0},
    "Code2003-egdm" : {'ext':".ttf",'font_size':140,'adjust':0.21},
    "NotoSansEgyptianHieroglyphs-Regular" : {'ext':".ttf",'font_size':100,'adjust':0.21},
    "NewGardinerSMP" : {'ext':".ttf",'font_size':190,'adjust':0},
    "seguihis" : {'ext':".ttf",'font_size':140,'adjust':0.3},
    "SINUHE" : {'ext':".ttf",'font_size':190,'adjust':0},
    "AbydosB" : {'ext':".ttf",'font_size':140,'adjust':0},
    "AbydosR" : {'ext':".ttf",'font_size':140,'adjust':0}, 
    "AaronBasicRTLAlpha100" : {'ext':".ttf",'font_size':190,'adjust':0},    
    "AegyptusBold" : {'ext':".otf",'font_size':190,'adjust':0},
    "Code2003-W8nn" : {'ext':".ttf",'font_size':140,'adjust':0.21},
}