import text
from utils import load_filepaths_and_text

filelists = ["filelists/newcombine_test_filelist.txt", "filelists/newcombine_train_filelist.txt", "filelists/newcombine_val_filelist.txt"]
out_extension = "cleaned"
text_index = 2

for filelist in filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in range(len(filepaths_and_text)):
       original_text = filepaths_and_text[i][text_index]
       clearner = ['english_cleaners']
       if filepaths_and_text[i][1] == '2':
           clearner[0] = 'universal_cleaners'
       elif filepaths_and_text[i][1] == '3':
           clearner[0] = 'japanese_cleaners'
       elif filepaths_and_text[i][1] == '4':
           clearner[0] = 'korean_cleaners'

       cleaned_text = text._clean_text(original_text, clearner)
       filepaths_and_text[i][text_index] = cleaned_text

    new_filelist = filelist + "." + out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
        f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])