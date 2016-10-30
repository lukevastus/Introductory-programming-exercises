import webbrowser
import numpy as np
import re
from tkinter import *


def open_coursedes(major):
    if major.upper() in majors.values():
        webbrowser.open("http://www.registrar.ucla.edu/Academics/Course-Descriptions/Course-Details?SA=" +
                        convert_symbol(major) + "&funsel=3")
    elif major.lower() in majors.keys():
        webbrowser.open(
            "http://www.registrar.ucla.edu/Academics/Course-Descriptions/Course-Details?SA=" +
            convert_symbol(majors[major.lower()]) + "&funsel=3")
    else:
        raise ValueError("Subject area not found")

def convert_symbol(string):
    for i in range(len(string)):
        if string[i] == "&":
            return string[0:i] + "%26" + string[i + 1:]
    return string

with open("Majors.txt", "r") as myfile:
    text = myfile.readlines()
majors = {}
for lines in text:
    name = re.search(r"(?<=\t)[A-Z][a-z]+"
                     r"(( and| of| in| as a| \|| the|, Study of)*"
                     r"( |(\/)|-|, )[A-Z][a-z,]+)*"
                     r"( \(undergraduate\))*"
                     r"( \(Graduate\))*"
                     r"( \(pre-16F\))*"
                     r"( \(pre-15F\))*"
                     r"(?=\t|  )", lines)
    abbrev = re.search(r"(?<=\t)([A-Z -\/&])+(?=(\t| )"
                       r"(AA|DN|EI|EN|GS|HU|IS|LF|LW|MG|MN|MU|NS|PA|PH|PS|SM|SS|TF)\t)", lines)
    if name and abbrev:
        majors.update({name.group(0).lower() : abbrev.group(0)})

majors.update({"religion, study of": "RELIGN", "neuroscience": "NEUROSC",
               "mechanical and aerospace engineering": "MECH&AE"})

#with open("Subject areas compiled.txt", "w") as myfile:
    #for keys, values in majors.items():
        #myfile.write(keys + ": " + values + "\n")

root = Tk()

root.title("Course descriptions finder")

label = Label(root, text="Enter subject area name or abbreviation:")
label.pack()

entry = Entry(root)
entry.pack()

open_des = Button(root, text="Open course description website", command=lambda: open_coursedes(entry.get()))
open_des.pack()

open_myucla = Button(root, text="MyUCLA", command=lambda: webbrowser.open("https://my.ucla.edu/"))
open_myucla.pack()

open_ccle = Button(root, text="CCLE", command=lambda: webbrowser.open("https://ccle.ucla.edu/"))
open_ccle.pack()

root.mainloop()
