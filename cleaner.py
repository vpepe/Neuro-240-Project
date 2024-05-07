import re as regex
import matplotlib.pyplot as plt

diachronica = "diachronica-data-1.csv"
target = "diachronica-trimmed.tsv"
pattern = '''[^'"\t\[\]${}\(\)|ːVCN+_ʼ1234567890ʷ~ˤˀʰʱˑ@ᵑⁿᵐ?ʲ%”…\*\.-]+'''
disallowed = ["voiced","lateral","palatal","coronal","dorsal","unvoiced","∅","#","from","to","tone","falling","rising","voice","long","low","high","pitch","mid","falling\xa0tone"]

with open(f"{diachronica}","r", encoding='utf8') as db, open(f"{target}","w+", encoding='utf8') as tgt:
    for line in db:
        args = line.split(',')
        source = regex.findall(pattern,args[-3])
        destination = regex.findall(pattern,args[-2])

        if source == [] or destination == [] or any([i.strip().lower() in disallowed for i in source]) or any([i.strip().lower() in disallowed for i in destination]) or source == destination:
            continue
        

        if len(source) > 1 or len(destination) > 1:
            if len(source) == len(destination):
                for i in range(len(source)):
                    tgt.write(f"{source[i].lower()}\t{destination[i].lower()}\n")
            if len(source) > len(destination):
                for i in range(len(source)):
                    for j in range(len(destination)):
                        tgt.write(f"{source[i].lower()}\t{destination[j].lower()}\n")
            if len(source) < len(destination):
                for i in range(len(destination)):
                    for j in range(len(source)):
                        tgt.write(f"{source[j].lower()}\t{destination[i].lower()}\n")
                pass
        else:
            tgt.write(f"{source[0].lower()}\t{destination[0].lower()}\n")

with open(f"{target}","r", encoding='utf8') as db:
    graphDict = {}
    for line in db:
        change = line.split("\t")
        start, end = change[0].strip(), change[1].strip()
        try:
            graphDict[start][end] += 1
        except:
            try:
                graphDict[start][end] = 1
            except:
                graphDict[start] = {}

overallDict = {key: sum(graphDict[key].values()) for key in graphDict.keys() if sum(graphDict[key].values()) >= 15}

acceptable_phones = ['u','a','i',"e","o","ɛ",'ɔ',"ə","ɪ","ʊ","ɑ","æ","ɯ","y","q",'k',"g",'p',"b","t","d","ʈ","c",'m',"n",'ŋ',"ɲ","ɳ","β","ɸ","ð","χ","f","v","s","z",'ʒ','ʃ',"x",'ɣ',"ʁ",'j',"r","ɾ","ɽ","ɭ",'ʔ','w',"l","h"]

graphDictKeys, graphDictClean = {key: graphDict[key] for key in acceptable_phones}, {}

for sourceKey,destinations in graphDictKeys.items():
    graphDictClean[sourceKey] = {}
    for destinationKey, destinationCount in destinations.items():
        if destinationKey in acceptable_phones:
            graphDictClean[sourceKey][destinationKey] = destinationCount
        else:
            continue
    for i in acceptable_phones:
        try:
            graphDictClean[sourceKey][i]
        except: 
            graphDictClean[sourceKey][i] = 0
    graphDictClean[sourceKey] = dict(sorted(graphDictClean[sourceKey].items()))

list_1 = []
for i in graphDictClean.keys():
    graphDictClean[i][i] = (sum(graphDict[i].values()) - graphDictClean[i][i])
    denom = graphDictClean[i][i]
    for j in graphDictClean.keys():
        graphDictClean[i][j] = graphDictClean[i][j] / denom
        list_1.append(graphDictClean[i][j])
        if graphDictClean[i][j] >= 0.1:
            graphDictClean[i][j] = 1
        if graphDictClean[i][j] < 0.1:
            graphDictClean[i][j] = 0