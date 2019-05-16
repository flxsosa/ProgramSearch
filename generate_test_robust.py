import random

# ========================== ADDRESS ==============================
def street_number():
    return random.choice([_ for _ in range(400)])

def street():
    return random.choice(['Pine', "Brook", "Foothill", "Broadway"])

def street_type():
    return random.choice(["Street", "Road", "Way", "Ave", "ST"])

def street_zip():
    return random.choice( range(10000, 100000))

def street_state():
    return random.choice(['CA','PA','IL'])

def address_info():
    return street_number(), street(), street_type(), street_zip(), street_state()

def street_ex1(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num} {st} {st_type} {st_zip}", f"Num:{st_num} Street:{st}"]
def street_ex2(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num}, {st} {st_type}, {st_state}", f"{st} {st_type} ({st_state})"]
def street_ex3(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num}, {st} {st_type}, {st_state}, {st_zip}", f"{st_state},{st_zip}"]
def street_ex4(st_num, st, st_type, st_zip, st_state):
    zip_suffix = street_zip()
    return [f"{st_num}, {st} {st_type}, {st_state}, {st_zip}-{zip_suffix}", f"{st_state} {st_zip}"]

def street_ex(n_io):
    infos = [address_info() for _ in range(n_io)]
    ex_kind = random.choice([street_ex1, street_ex2, street_ex3, street_ex4])
    return [ex_kind(*info) for info in infos]

# ========================= NAMES ============================
def names_first():
    return random.choice(["Alex", "Sasha", "Taylor", "Jackie"])

def names_last():
    return random.choice(["Lennon", "Smith", "Einstein", "Schmidhuber"])

def names_middle():
    return random.choice([names_last(), ""])

def names_title():
    return random.choice(["Dr", "Sir", "Mr", "Mrs"])

def names_suffix():
    return random.choice([random.choice(["Esq", "I", "II", "Jr"]), ""])

def name_info():
    return names_first(), names_last(), names_middle(), names_title(), names_suffix()

def name_ex1(first, last, middle, title, suffix):
    return [f"{title} {first} {middle} {last} {suffix}", f"{first} : {last}"]
def name_ex2(first, last, middle, title, suffix):
    return [f"{first} {middle} {last}", f"Dr {last}"]
def name_ex3(first, last, middle, title, suffix):
    return [f"{title} {first} {middle} {last}", f"{last}, {first} ({title})"]
def name_ex4(first, last, middle, title, suffix):
    last_letter = last[0]
    first_letter = first[0]
    return [f"{title} {first} {middle} {last}", f"{last}, {first_letter})"]

def name_ex(n_io):
    infos = [name_info() for _ in range(n_io)]
    ex_kind = random.choice([name_ex1, name_ex2, name_ex3, name_ex4])
    return [ex_kind(*info) for info in infos]

if __name__ == '__main__':
    print (street_ex(4))
    print (name_ex(4))
