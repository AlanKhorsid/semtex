def parse_string(inputString:str) -> [str]:
    listedCharacters = list(inputString)
    outputStrings = []
    string = ""
    quoteCounter = 0
    for c in listedCharacters:
        if c == "\"" and quoteCounter == 0:
            string = ""
            string = string + c
            quoteCounter += 1
        elif c == "\"" and quoteCounter == 1:
            string = string + c
            quoteCounter -= 1
        elif c == ","  and quoteCounter == 0:
            outputStrings.append(string)
            string = ""
        else:
            string = string + c
    outputStrings.append(string)
    if quoteCounter == 1:
        raise Exception('Uneven amount of quotation marks')

    return outputStrings

print(parse_string("\"Uniastate, Bears\",1,2,\"hi, you\", yes"))
print(parse_string("\"awmaod"))
