numberList = []

while True:
    numInput = input("Unesi broj ili 'Done' za zavrsetak: ")
    if numInput == "Done" : 
        break
    try:
        numInput = float(numInput)
        numberList.append(numInput)
    except:
        print("Pogresan unos")

avg = sum(numberList) / len(numberList)

if numberList:
    print(len(numberList))
    print(avg)
    print(min(numberList))
    print(max(numberList))
    print(sorted(numberList))