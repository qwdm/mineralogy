#!/usr/bin/env python3

import os
import sys
import os.path
import argparse
# import matplotlib.pyplot as plt
# import numpy as np

from typing import List, Optional
from math import ceil
from decimal import Decimal
from time import time
from datetime import datetime
from dataclasses import dataclass, field

LINENUM_DATE = 1  # в python считается с 0, поэтому по сравнению с номером в файле смещение на 1
LINENUM_TIME = 2
LINENUM_TITLE = 3  # TODO ??? или ЯРЛЫК ПОКАЗ, ???

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


@dataclass
class Spectrum:
    title: str
    date: str 
    time: str
    olympus_spectrum: List[int]

    def __post_init__(self):
        self._set_date(self.date)
        self._set_time(self.time)
        self.equalized_spectrum = self._equalize_scales(self.olympus_spectrum)

    def _set_date(self, date_str : str):
        """
        Получает дату в формате 2023-04-05
        Устанавливает дату в формате 05-APR-2023
        Если входной формат не подошел, то просто берем что получили
        """

        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            self.date = date_obj.strftime("%d-%b-%Y").upper()
        except ValueError:
            self.date = date_str

    def _set_time(self, time_str : str):
        """
        Получает время в формате 13:31:49
        Устанавливает время в формате 13:31:49
        Если входной формат другой, то устанавливаем что получили
        """

        time_elements = time_str.split(':')
        if len(time_elements) == 3:
            self.time = ':'.join(time_elements[:2])  # оставим только часы и минуты
        else:
            self.time = time_str

    @staticmethod
    def _equalize_scales(olympus_spectrum : List[int]) -> List[int]:
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


@dataclass
class InputFile:
    """
    Извлекает из входного файла списки из колонок данных
    Основной метод get_spectrums Возвращает список объектов типа Spectrum
    """
    filename: str

    titles: List[str] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)
    spectrums: List[List[int]] = field(default_factory=list)

    _data_names = ['titles', 'dates', 'times', 'spectrums']

    def read(self) -> List[str]:
        self.lines = []
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

                self.lines.append(line)
        
        return self.lines

    @staticmethod
    def _get_columns(spectrum_data : List[List[int]]) -> List[List[int]]:
        """
        Транспонируем спектральные данные, образовав
        строчки спектров из колонок.

        Сделаем это вручную, чтобы не зависеть от numpy
        """
        col_spectrums = [[] for _ in spectrum_data[0]]

        for line in spectrum_data:
            for i, elem in enumerate(line):
                col_spectrums[i].append(elem)
        
        return col_spectrums

    def _extract_spectrums(self) -> List[List[int]]:
        """
        Извлечь данные о спектрах из строк файла
        """
        spectrum_data = []

        # Уберем слово "Данные" из первой строки спектральных данных
        if not self.lines[DATA_START_LINE].startswith('Данные'):
            raise Exception("Начало данных отсутсвует или смещено")
        spectrum_data.append([int(value) for value in self.lines[DATA_START_LINE].split(',')[1:]])

        cnt = 1
        for line in self.lines[DATA_START_LINE + 1:]:
            if  len(line) > 1:  # только непустые строки с данными
                spectrum_data.append([int(value) for value in line.split(',')])
                cnt +=1

        if cnt != DATA_LINES_AMOUNT:
            raise Exception(f"Прочитано {cnt} строк спектра; ожидалось {DATA_LINES_AMOUNT} строк.")

        self._spectrum_data = spectrum_data
        self.spectrums = self._get_columns(spectrum_data)

        return self.spectrums

    @staticmethod
    def _extract_info_line(line) -> List[str]:
        """
        Отбрасывает первый элемент -- имя строки
        Возвращает список данных по строке
        """
        return line.split(',')[1:]  # отбрасываем имя строки


    def _extract_info(self):
        self.titles = self._extract_info_line(self.lines[LINENUM_TITLE])
        self.dates = self._extract_info_line(self.lines[LINENUM_DATE])
        self.times = self._extract_info_line(self.lines[LINENUM_TIME])

    def _cnt_and_check(self) -> int:
        """ 
        Подсчитывает количество столбцов со спектрами, 
        проверяя что во всех строках их одинаковое количество.
        """
        cnt = len(self.spectrums)
        for name in self._data_names:
            attrlist = getattr(self, name)
            if len(attrlist) != cnt:
                raise Exception("Строки данных имеют различное число столбцов")

        return cnt

    def get_spectrums(self) -> List[Spectrum]:

        self._extract_spectrums()
        self._extract_info()

        self._cnt_and_check()

        spectrum_objs = []

        for i, _ in enumerate(self.spectrums):
            spectrum_objs.append(Spectrum(
                title=self.titles[i],
                date=self.dates[i],
                time=self.times[i],
                olympus_spectrum=self.spectrums[i],
            ))
        
        return spectrum_objs


@dataclass
class OutFile:
    filename : str

    @staticmethod
    def _make_csv_spectrum(spectrum : Spectrum) -> List[str]:
        """
        Принимает значения спектра, возвращает пары строк в требуемом формате
        энергия, число_импульсов.

        например,
        19.45, 623.
        """
        energy = Decimal('-0.20')
        delta = Decimal('0.01')

        csv_spectrum = []

        for value in spectrum.equalized_spectrum:
            csv_spectrum.append(f"{energy}, {value}.")
            energy += delta
        
        return csv_spectrum

    def write(self, spectrum : Spectrum) -> None:
        """Создает один файл по одной из предобработанных колонок"""
        csv_spectrum = self._make_csv_spectrum(spectrum)

        data = OUTPUT_FILE_HEADER.format(
            date=spectrum.date,
            time=spectrum.time,
            title=spectrum.title,
        )
        data += '\n'.join(csv_spectrum)
        data += '\n#ENDOFDATA   :\n'

        with open(self.filename, 'w') as out:
            out.write(data)




def create_files(
        spectrums : List[Spectrum],
        input_filename : str,
        asked_titles : Optional[List[str]] = None) -> None:
    """
    Создает файлы по всем запрошенным (asked) именам, 
    или по вообще всем, если не был указан список имен 
    """

    if asked_titles is None:
        print("Все спектры будут обработаны")
        asked_titles = [s.title for s in spectrums]

    asked_titles = set(asked_titles)

    _, input_filename = os.path.split(input_filename)
    input_prefix, _ = os.path.splitext(input_filename)

    dirname = f"{DIRNAME}_{input_prefix}_{int(time())}"  # добавим целочисленный timestamp в имя файла
                                                         # для уникальности и последовательности
    os.mkdir(dirname)

    for spectrum in spectrums:
        if spectrum.title in asked_titles:
            filename = os.path.join(dirname, f"{spectrum.title}.txt")
            outfile = OutFile(filename)
            outfile.write(spectrum)
            asked_titles.remove(spectrum.title)
        else:
            pass
    
    # после всей обработки не должно остаться имен в очереди запросов,
    # иначе этих имен вовсе не было в файле
    if asked_titles:
        print("Не найдены имена: ", ', '.join(asked_titles))
    

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
        "-t", "--titles", nargs="+",
        help="Имена спектров для обработки. Ненайденные имена после обработки будут выведены."
    )
    args = parser.parse_args()

    ##################################
    #### КОНЕЦ ОПИСАНИЯ ИНТЕРФЕЙСА ###
    ##################################

    ## Если не указано имя файла, обрабатываем все csv в директории
    if args.input is not None:
        filenames = [args.input]
    else:
        filenames = []
        for filename in os.listdir():
            _, ext = os.path.splitext(filename)
            if ext == '.csv':
                filenames.append(filename)
    
    for filename in filenames:

        print(f"Oбрабатывается: {filename}")

        infile = InputFile(filename)
        try:
            infile.read()
        except FileNotFoundError:
            print("Файла '{filename}' не найдено", file=sys.stderr)
            continue

        spectrums : List[Spectrum] = infile.get_spectrums()

        ##------------------------------------------------------------------##
        create_files(spectrums, 
                     input_filename=filename, 
                     asked_titles=args.titles)
        ##------------------------------------------------------------------##

        # выключим эту опцию для независимости от pyplot
        # Если включена опция построения графика для визуального контроля, то построим 
        # if args.graph:
        #     equalized = equalize_scales(col_spectrums[0])

        #     # print(len(equalized))
        #     plt.plot(np.log(equalized))
        #     plt.show()