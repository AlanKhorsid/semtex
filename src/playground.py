# def clean_string(input_str):
#     cleaned_str = input_str.replace("''", '"').replace('""', '"')
#     parts = cleaned_str.split('"')

#     result = []
#     for i, part in enumerate(parts):
#         if i % 2 == 0:
#             result.append(part.strip())
#         else:
#             result.append(part)

#     cleaned_str = " ".join(result).strip()
#     return cleaned_str


# input_string = """"Moth"" Pendat and Box", "East""Riding of Yorkshire"."""
# cleaned_string = clean_string(input_string)
# print(cleaned_string)

import html


print(html.unescape('East"Riding of Yorkshire'))
"East" "Riding of Yorkshire"
"East" "Riding of Yorkshire"

original_string = '"Hammer and Sickle" gold medal'
standardized_string = original_string.replace('"', '""').replace("\\", "")

print(standardized_string)  # """Hammer and Sickle"" gold medal"
