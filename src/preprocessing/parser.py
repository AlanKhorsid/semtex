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
            outputStrings.append(string)
            string = ""
            quoteCounter -= 1
        elif c == ","  and quoteCounter == 0:
            outputStrings.append(string)
            string = ""
        else:
            string = string + c
    if quoteCounter == 1:
        print("ERROR UNEVEN AMOUNT OF QUOTATION MARKS")

    return outputStrings

print(parse_string("hi, \"hello,greetings\",5,6"))
