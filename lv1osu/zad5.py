def avgWordCount(smsList):
    totalWords = 0
    for sms in smsList:
        totalWords += len(sms.split(" "))
    return totalWords / len(smsList)

def endsOnExlMark(sms):
    return sms[-1] == '!'

ham = []
spam = []

smsFile = open("SMSSpamCollection.txt")
for line in smsFile:
    line = line.rstrip()
    parts = line.split("\t")
    if (parts[0] == "ham"):
        ham.append(parts[1])
    elif (parts[0] == "spam"):
        spam.append(parts[1])

print(f"Prosjecan broj rijeci u ham: {avgWordCount(ham)}")
print(f"Prosjecan broj rijeci u spam: {avgWordCount(spam)}")
print(f"Broj rijeci koje zavrsavaju usklicnikom: {len(list(filter(endsOnExlMark, spam)))}")