try:
    ocjena = float(input("Unesi ocjenu: "))
    if 0.0 < ocjena < 1.0 :
        if ocjena >= 0.9 : 
            print("A")
        elif ocjena >= 0.8 : 
            print("B")
        elif ocjena >= 0.7 : 
            print("C")
        elif ocjena >= 0.6 : 
            print("D")
        else: 
            print("F")
except:
    print("Pogresan unos. Ocjena mora biti izmedu 0.0 i 1.0")