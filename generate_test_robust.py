import random

def street_number():
    return random.choice([_ for _ in range(400)])

def street():
    return random.choice(['Pine', "Brook", "Vassar",])

def street_type():
    return random.choice(["Street", "Road", "Way", "Ave"])

def street_zip():
    return random.choice( range(10000, 100000))

def street_state():
    return random.choice(['CA','PA'])

def address_info():
    return street_number(), street(), street_type(), street_zip(), street_state()

def street_ex1(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num} {st} {st_type} {st_zip}", f"Num:{st_num} Street:{st}"]
def street_ex2(st_num, st, st_type, st_zip, st_state):
    return [f"{st_num},{st},{st_type},{st_state}", f"{st} {st_type} ({st_state})"]

def street_ex(n_io):
    infos = [address_info() for _ in range(n_io)]
    ex_kind = random.choice([street_ex1, street_ex2])
    return [ex_kind(*info) for info in infos]

if __name__ == '__main__':
    print (street_ex(4))
