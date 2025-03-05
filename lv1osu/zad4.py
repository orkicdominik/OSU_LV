wordCount = {}

songFile = open("song.txt")
for line in songFile:
    line = line.rstrip()
    words = line.split(" ")
    for word in words:
        if word not in wordCount:
            wordCount[word] = 1
            continue
        wordCount[word] = wordCount[word] + 1
songFile.close()
uniqueWords = 0
for word in wordCount:
    if wordCount[word] == 1:
        uniqueWords += 1
print("{word} : {wordCount[word]}")
print(uniqueWords)