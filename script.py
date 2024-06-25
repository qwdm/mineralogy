#!/usr/bin/env python3

import os
import sys
import os.path
import argparse
# import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from math import ceil
from decimal import Decimal
from time import time


DATA_START_LINE = 40
DATA_LINES_AMOUNT = 2048
DIRNAME = "out"

DEFAULT_DATE = "25-AUG-2016"
DEFAULT_TIME = "11:24"
DEFAULT_TITLE = "1"

OUTPUT_FILE_HEADER = \
"""#FORMAT      : EMSA/MAS Spectral Data File
#VERSION     : 1.0
#TITLE       : Спектр {title}
#DATE        : {date}
#TIME        : {time}
#OWNER       : 
#NPOINTS     : 2048.
#NCOLUMNS    : 1.
#XUNITS      : keV
#YUNITS      : counts
#DATATYPE    : XY
#XPERCHAN    : 0.0100000
#OFFSET      : -0.200000
#SIGNALTYPE  : EDS
#CHOFFSET    : 20.0000
#LIVETIME    : 120.00000
#REALTIME    : 138.26053
#BEAMKV      : 20.0000
#PROBECUR    : 0.000000
#MAGCAM      : 8161.00
#XTILTSTGE   : 0.0
#AZIMANGLE   : 0.0
#ELEVANGLE   : 35.0
#XPOSITION mm: 0.0000
#YPOSITION mm: 0.0000
#ZPOSITION mm: 0.0000
##OXINSTPT   : 5
##OXINSTSTROB: 40.15
##OXINSTELEMS: 6,92
##OXINSTLABEL: 6, 0.277, C
##OXINSTLABEL: 92, 13.615, U
##OXINSTLABEL: 92, 17.220, U
##OXINSTLABEL: 92, 16.429, U
##OXINSTLABEL: 92, 15.400, U
##OXINSTLABEL: 92, 11.618, U
##OXINSTLABEL: 92, 3.171, U
##OXINSTLABEL: 92, 3.337, U
##OXINSTLABEL: 92, 3.564, U
##OXINSTLABEL: 92, 2.455, U
##OXINSTLABEL: 92, 4.401, U
#SPECTRUM    : Spectral Data Starts Here
"""



def read_file(filename: str) -> List[str]:
    lines = []
    with open(filename) as f:
        for line in f:
            # уберем перевод строки в конце
            # и начальный и конечный символ запятой
            line = line.rstrip() 

            # только непустые строки
            if len(line) < 1:
                continue

            if line[0] == ',':
                line = line[1:]
            if line[-1] == ',':
                line = line[:-1]

            lines.append(line)
    
    return lines


def extract_spectrums(lines : List[str]) -> List[List[int]]:
    spectrum_data = []

    # Уберем слово "Данные" из первой строки спектральных данных
    if not lines[DATA_START_LINE].startswith('Данные'):
        raise Exception("Начало данных отсутсвует или смещено")
    spectrum_data.append([int(value) for value in lines[DATA_START_LINE].split(',')[1:]])

    cnt = 1
    for line in lines[DATA_START_LINE + 1:]:
        if  len(line) > 1:  # только непустые строки с данными
            spectrum_data.append([int(value) for value in line.split(',')])
            cnt +=1

    if cnt != DATA_LINES_AMOUNT:
        raise Exception(f"Прочитано {cnt} строк спектра; ожидалось {DATA_LINES_AMOUNT} строк.")

    return spectrum_data


def cnt_and_check_spectrums(spectrum_data : List[List[int]]) -> int:
    """ 
    Подсчитывает количество столбцов со спектрами, 
    проверяя что во всех строках их одинаковое количество.
    """
    if len(set([len(line) for line in spectrum_data])) != 1:
        raise Exception("Строки данных имеют различное число столбцов")

    num_cols = len(spectrum_data[0])

    return num_cols


def get_columns(spectrum_data : List[List[int]]) -> np.array:
    """Транспонируем исходные данные,
    образовав спектры из колонок"""

    col_spectrums = np.array(spectrum_data)
    col_spectrums = col_spectrums.T
    return col_spectrums


def equalize_scales(olympus_spectrum : List[int]) -> List[int]:
    """
    Уравнивает шкалы. Пересчет из измеренных данных спектра в данные для обработки
    """
    equalized_spectrum = [0 for _ in range(20)]
    
    # всего в выходном файле 2048 строк. 
    # 20 из них сразу заполнены нулями
    # между каждыми двумя строками и в последнюю строку
    # вставлены средние соседей. Поэтому осталось взять
    # из реальных измерений (2048 - 20) / 2 = 1014 строк.
    # самая последняя строка пересчитанного спектра
    # берется средним между 0 и последней строкой диапазона
    # из измеренного спектра 

    for i in range(1014):
        equalized_spectrum.append(olympus_spectrum[i])
        equalized_spectrum.append(ceil((olympus_spectrum[i] + olympus_spectrum[i + 1]) * .5))  # среднее 
    
    # последнее среднее между нулем и предпоследним. Т.е. половина предпоследнего
    # Изменим это значение.
    equalized_spectrum[-1] = ceil((olympus_spectrum[i] * .5))

    return equalized_spectrum


def make_csv_spectrum(equalized_spectrum : List[int]) -> List[str]:
    """
    Принимает значения спектра, возвращает пары строк в требуемом формате
    энергия, число_импульсов.

    например,
    19.45, 623.
    """

    energy = Decimal('-0.20')
    delta = Decimal('0.01')

    csv_spectrum = []

    for value in equalized_spectrum:
        csv_spectrum.append(f"{energy}, {value}.")
        energy += delta
    
    return csv_spectrum


def make_outfile(csv_spectrum : List[str], filename : str, info : dict) -> None:
    """Создает один файл по одной из предобработанных колонок"""
    data = OUTPUT_FILE_HEADER.format(
        date=info.get('date', DEFAULT_DATE),
        time=info.get('time', DEFAULT_TIME),
        title=info.get('title', DEFAULT_TITLE),
    )
    data += '\n'.join(csv_spectrum)
    data += '\n#ENDOFDATA   :\n'

    with open(filename, 'w') as out:
        out.write(data)


def extract_spectrum_names(file_lines) -> List[str]:
    """Из 4 строки файла берем имена спектров. В питоне строки считаются с 0, поэтому 3"""
    line = file_lines[3]
    names = line.split(',')[1:]  # отбрасываем имя строки
    return names


def create_files(
        col_spectrums : np.array,
        spectrum_names : List[str],
        input_filename : str,
        asked_names : Optional[List[str]] = None) -> None:
    """
    Создает файлы по всем запрошенным (asked) именам, 
    или по вообще всем, если не был указан список имен 
    """

    if asked_names is None:
        print("Все спектры будут обработаны")
        asked_names = spectrum_names[:]

    asked_names = set(asked_names)

    _, input_filename = os.path.split(input_filename)
    input_prefix, _ = os.path.splitext(input_filename)

    dirname = f"{DIRNAME}_{input_prefix}_{int(time())}"  # добавим целочисленный timestamp в имя файла
                                                         # для уникальности и последовательности
    os.mkdir(dirname)

    for i, spectrum in enumerate(col_spectrums):
        if spectrum_names[i] in asked_names:
            equalized = equalize_scales(spectrum)
            csv_spectrum = make_csv_spectrum(equalized)
            filename = os.path.join(dirname, f"{spectrum_names[i]}.txt")
            make_outfile(csv_spectrum, filename)
            asked_names.remove(spectrum_names[i])
        else:
            pass
    
    # после всей обработки не должно остаться имен в очереди запросов,
    # иначе этих имен вовсе не было в файле
    if asked_names:
        print("Не найдены имена: ", ', '.join(asked_names))
    

if __name__ == '__main__':

    ###################################
    #### ИТЕРФЕЙС КОМАНДНОЙ СТРОКИ ####
    ###################################

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str,
        help="Входной файл csv, по умолчанию обрабатываются все имеющиеся csv в текущей директории"
    )
    # parser.add_argument(
    #     "-g", "--graph", action="store_true",
    #     help="Показать для визуального контроля график первого столбца спектра после обработки"
    # )
    parser.add_argument(
        "-n", "--names", nargs="+",
        help="Имена спектров для обработки. Ненайденные имена после обработки будут выведены."
    )
    args = parser.parse_args()

    ##################################
    #### КОНЕЦ ОПИСАНИЯ ИНТЕРФЕЙСА ###
    ##################################


    # TODO get filename from user / search default
    # filename = "beamspectra-842739-2023-04-06-11-53-20.csv"

    if args.input is not None:
        filenames = [args.input]
    else:
        filenames = []
        for filename in os.listdir():
            _, ext = os.path.splitext(filename)
            if ext == '.csv':
                filenames.append(filename)
    
    print("DEBUG:", filenames)

    for filename in filenames:

        print(f"Oбрабатывается: {filename}")

        try:
            file_lines = read_file(filename)
        except FileNotFoundError:
            print("Файла '{filename}' не найдено", file=sys.stderr)
            continue

        spectrum_names = extract_spectrum_names(file_lines)
        spectrum_data = extract_spectrums(file_lines)

        num_cols = cnt_and_check_spectrums(spectrum_data)
        if num_cols != len(spectrum_names):
            raise Exception("Количество спектров не равно количеству имен спектров")

        col_spectrums = get_columns(spectrum_data)

        ##------------------------------------------------------------------##
        create_files(col_spectrums, spectrum_names, input_filename=filename, asked_names=args.names)
        ##------------------------------------------------------------------##

        # Если включена опция построения графика для визуального контроля, то построим 
        # if args.graph:
        #     equalized = equalize_scales(col_spectrums[0])

        #     # print(len(equalized))
        #     plt.plot(np.log(equalized))
        #     plt.show()