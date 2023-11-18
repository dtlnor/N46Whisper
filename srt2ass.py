# -*- coding: utf-8 -*-
#
# python-srt2ass: https://github.com/ewwink/python-srt2ass
# by: ewwink
# modified by:  一堂宁宁 Lenshyuu227
# modified by:  dtlnor, 2023-10-16

import sys
import os
import re
import codecs


def fileopen(input_file):
    # use correct codec to encode the input file
    encodings = ["utf-32", "utf-16", "utf-8", "cp1252", "gb2312", "gbk", "big5"]
    srt_src = ''
    for enc in encodings:
        try:
            with codecs.open(input_file, mode="r", encoding=enc) as fd:
                # return an instance of StreamReaderWriter
                srt_src = fd.read()
                break
        except:
            # print enc + ' failed'
            continue
    return [srt_src, enc]


def srt2ass(input_file, is_split = "No", split_method = "Modest", split_req = 5):
    '''
    Parameters:
        input_file: srt to process.

        is_split: 将存在空格的单行文本分割为多行（多句）。
            分割后的若干行均临时采用相同时间戳，
            且添加了adjust_required标记提示调整时间戳避免叠轴。

        split_method: 
            普通分割 (Modest): 
                当空格后的文本长度超过 split_req 个字符，则另起一行
            全部分割 (Aggressive): 
                只要遇到空格即另起一行

        split_req: 当空格后的文本长度超过 n 个字符，则另起一行
    '''

    if '.ass' in input_file:
        return input_file

    if not os.path.isfile(input_file):
        print(input_file + ' not exist')
        return

    src = fileopen(input_file)
    srt_content = src[0]
    # encoding = src[1] # Will not encode so do not need to pass codec para
    src = ''
    utf8bom = ''

    if u'\ufeff' in srt_content:
        srt_content = srt_content.replace(u'\ufeff', '')
        utf8bom = u'\ufeff'
    
    srt_content = srt_content.replace("\r", "")
    lines = [x.strip() for x in srt_content.split("\n") if x.strip()]
    subLines = ''
    dlgLines = '' # dialogue line
    lineCount = 0
    output_file = '.'.join(input_file.split('.')[:-1])
    output_file += '.ass'

    for ln in range(len(lines)):
        line = lines[ln]
        if line.isdigit() and re.match('-?\d\d:\d\d:\d\d', lines[(ln+1)]):
            if dlgLines:
                subLines += dlgLines + "\n"
            dlgLines = ''
            lineCount = 0
            continue
        else:
            if re.match('-?\d\d:\d\d:\d\d', line):
                line = line.replace('-0', '0')
                dlgLines += 'Dialogue: 0,' + line + ',Default,,0,0,0,,'
            else:
                if lineCount < 2:
                    dlg_string = line
                    if is_split == "Yes" and split_method == 'Modest':
                        # do not split if space proceed and followed by non-ASC-II characters
                        # do not split if space followed by word that less than 5 characters
                        split_string = re.sub(r'(?<=[^\x00-\x7F])\s+(?=[^\x00-\x7F])(?=\w{'+split_req+r'})', r'|', dlg_string)
                        # print(split_string)
                        if len(split_string.split('|')) > 1:
                            dlgLines += (split_string.replace('|', "(adjust_required)\n" + dlgLines)) + "(adjust_required)"
                        else:
                            dlgLines += line
                    elif is_split == "Yes" and split_method == 'Aggressive':
                        # do not split if space proceed and followed by non-ASC-II characters
                        # split at all the rest spaces
                        split_string = re.sub(r'(?<=[^\x00-\x7F])\s+(?=[^\x00-\x7F])', r'|', dlg_string)
                        if len(split_string.split('|')) > 1:
                            dlgLines += (split_string.replace('|',"(adjust_required)\n" + dlgLines)) + "(adjust_required)"
                        else:
                            dlgLines += line
                    else:
                        dlgLines += line
                else:
                    dlgLines += "\n" + line
            lineCount += 1
        ln += 1


    subLines += dlgLines + "\n"

    subLines = re.sub(r'\d(\d:\d{2}:\d{2}),(\d{2})\d', '\\1.\\2', subLines)
    subLines = re.sub(r'\s+-->\s+', ',', subLines)
    # replace style
    subLines = re.sub(r'<([ubi])>', "{\\\\\g<1>1}", subLines)
    subLines = re.sub(r'</([ubi])>', "{\\\\\g<1>0}", subLines)
    subLines = re.sub(r'<font\s+color="?#(\w{2})(\w{2})(\w{2})"?>', "{\\\\c&H\\3\\2\\1&}", subLines)
    subLines = re.sub(r'</font>', "", subLines)

    head_str = '''[Script Info]
; This is an Advanced Sub Station Alpha v4+ script.
Title:
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,2,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Actor, MarginL, MarginR, MarginV, Effect, Text'''
    
    output_str = utf8bom + head_str + '\n' + subLines
    # encode again for head string
    output_str = output_str.encode('utf8')

    with open(output_file, 'wb') as output:
        output.write(output_str)

    output_file = output_file.replace('\\', '\\\\')
    output_file = output_file.replace('/', '//')
    return output_file
