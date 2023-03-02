import pefile
import math
import datetime
import os
import hashlib


def get_md5(file_path):
    return hashlib.md5(open(file_path, 'rb').read()).hexdigest()


def extract_raw_integrated_features(pe_file, data):
    dos_header_fields = {'e_cp', 'e_cparhdr', 'e_maxalloc', 'e_sp', 'e_lfanew'}
    opt_header_fields = {'MajorLinkerVersion', 'MinorLinkerVersion', 'SizeOfCode',
                         'SizeOfInitializedData', 'SizeOfUninitializedData',
                         'AddressOfEntryPoint', 'BaseOfCode', 'BaseOfData',
                         'MajorOperatingSystemVersion', 'MinorOperatingSystemVersion',
                         'MajorImageVersion', 'MinorImageVersion', 'CheckSum',
                         'MajorSubsystemVersion', 'MinorSubsystemVersion', 'Subsystem',
                         'SizeOfStackReserve', 'SizeOfStackCommit', 'SizeOfHeapReserve',
                         'SizeOfHeapCommit', 'LoaderFlags'}

    for key, value in pe_file.DOS_HEADER.__field_offsets__.items():
        if key in dos_header_fields:
            data[key] = value
    for key, value in pe_file.OPTIONAL_HEADER.__field_offsets__.items():
        if key in opt_header_fields:
            data[key] = value
    try:
        data['e_cblp_indicator'] = int(pe_file.DOS_HEADER.e_cblp not in [80, 144])
    except AttributeError:
        data['e_cblp_indicator'] = 1

    # Fill missing values:
    for key in dos_header_fields - pe_file.DOS_HEADER.__field_offsets__.keys():
        data[key] = -1
    for key in opt_header_fields - pe_file.OPTIONAL_HEADER.__field_offsets__.keys():
        data[key] = -1

    warns = pe_file.get_warnings()
    data['has_warnings'] = 1 if warns else 0

    try:
        machine = pe_file.FILE_HEADER.Machine
        data['is_64bit'] = int(machine == 332)
    except AttributeError:
        data['is_64bit'] = 0

    try:
        sectionum = pe_file.FILE_HEADER.NumberOfSections
        data['NumberOfSections'] = sectionum
    except AttributeError:
        data['NumberOfSections'] = -1

    try:
        if pe_file.OPTIONAL_HEADER.DllCharacteristics == 0 and \
                pe_file.OPTIONAL_HEADER.MajorImageVersion == 0 and pe_file.OPTIONAL_HEADER.CheckSum == 0:
            data['article_indicator'] = 1
        else:
            data['article_indicator'] = 0
    except AttributeError:
        data['article_indicator'] = 1

    # Binarize Characteristics/DLLCharacteristics
    # If these fields are not present in a PE file, their values will be assumed to be zero by default.
    try:

        for i, bit in enumerate(bin(pe_file.FILE_HEADER.Characteristics)[2:].zfill(16)):
            data[f'FH_char{i}'] = bit
            if i == 14:
                break
    except AttributeError:
        for i in range(14):
            data[f'FH_char{i}'] = 0

    try:
        for i, bit in enumerate(bin(pe_file.OPTIONAL_HEADER.DllCharacteristics)[2:].zfill(12)):
            data[f'OH_DLLchar{i}'] = bit
            if i == 10:
                break
    except AttributeError:
        for i in range(10):
            data[f'OH_DLLchar{i}'] = 0


def pefile_entropy(data):
    if not data:
        return 0.0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy


def section_related_features(pe_file, data):
    sus_counter = 0
    legit_counter = 0
    special_counter = 0
    legit_sections = [".text", ".data", ".bss", ".rdata"]
    # As defined by the microsoft documentation about 'special sections'
    special_sections = [".debug", ".drective", ".edata", ".idata", ".pdata", ".reloc", ".tls", ".rsrc", ".cormeta",
                        ".sxdata"]
    text_ent = -1  # Assign default values incase of a missing section
    data_ent = -1
    current_min = 8
    current_max = -1
    for section in pe_file.sections:
        current_ent = section.get_entropy()
        current_max = current_ent if current_ent > current_max else current_max
        current_min = current_ent if current_ent > current_max else current_min
        sect_name = section.Name.decode().strip('\x00')
        if sect_name == '.text':
            text_ent = current_ent
        elif sect_name == '.data':
            data_ent = current_ent
        if sect_name in special_sections:
            legit_counter += 1
            special_counter += 1
        elif sect_name in legit_sections:
            legit_counter += 1
        else:
            sus_counter += 1
    data['MinSectionEntropy'] = current_min
    data['MaxSectionEntropy'] = current_max
    data['TotalSuspiciousSections'] = sus_counter
    data['TotalNonSuspiciousSections'] = legit_counter
    data['TotalSpecialSections'] = special_counter  # EXTRA FEATURE, could be useful
    data['.text_entropy'] = text_ent
    data['.data_entropy'] = data_ent


def file_related_features(pe_file, data):
    # Suspicious file creation year
    timestamp = pe_file.FILE_HEADER.TimeDateStamp
    cond = 0 if 1980 <= datetime.datetime.fromtimestamp(timestamp).year <= 2022 else 1
    data['SusCreationYear'] = cond

    # Checking whether the file has the FileInfo field:
    try:
        fileinfo = pe_file.FileInfo
        data['FileInfo'] = 1
    except AttributeError:
        data['FileInfo'] = 0

        # ImageBase default values- does not seem like its very informative but ok:
    try:
        imagebase = pe_file.OPTIONAL_HEADER.ImageBase
        cond = 1 if imagebase in [268435456, 65536, 4194304] or not imagebase % (64 * 1024) else 0
        data['ImageBase'] = cond
    except AttributeError:
        data['ImageBase'] = 0

    # Section alignment specification check
    try:
        filealign = pe_file.OPTIONAL_HEADER.FileAlignment
        sectionalign = pe_file.OPTIONAL_HEADER.SectionAlignment
        cond = 1 if sectionalign >= filealign else 0
        data['SectionAlignment'] = cond
        cond = 1 if math.log2(filealign).is_integer() and 512 <= filealign <= 65536 else 0
        data['FileAlignment'] = cond
    except AttributeError:
        data['SectionAlignment'] = 0
        data['FileAlignment'] = 0

    # Size of image and size of headers specification check
    try:
        sizeofimg = pe_file.OPTIONAL_HEADER.SizeOfImage
        sizeofheaders = pe_file.OPTIONAL_HEADER.SizeOfHeaders
        cond = 1 if not sizeofimg % sectionalign else 0
        data['SizeOfImage'] = cond
        cond = 1 if not sizeofheaders % filealign else 0
        data['SizeOfHeaders'] = cond
    except AttributeError:
        data['SizeOfImage'] = 0
        data['SizeOfHeaders'] = 0

    try:
        # EXTRA FEATURES - count of imported libraries and imported functions
        function_count = 0
        # Loop through each imported library and get the number of its imported functions
        libs = pe_file.DIRECTORY_ENTRY_IMPORT
        for lib in libs:
            function_count += len(lib.imports)
        data['ImportedLibsNum'] = len(libs)
        data['ImportedFuncsNum'] = function_count
    except:
        data['ImportedLibsNum'] = 0
        data['ImportedFuncsNum'] = 0

    # EXTRA FEATURE  - whether the file has a digital signature or not
    # ( https://snyk.io/advisor/python/pefile/functions/pefile.DIRECTORY_ENTRY )
    security_directory = pe_file.OPTIONAL_HEADER.DATA_DIRECTORY[
        pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_SECURITY']]
    is_signed = security_directory.VirtualAddress != 0
    data['DigitalSigned'] = int(is_signed)

    # Possible TODO: get hex of "the most important section"

    section_related_features(pe_file, data)


def path_related_features(pe_path, data):
    with open(pe_path, 'rb') as f:
        data_bin = f.read()  # Reading Bytes!
    ent = pefile_entropy(data_bin)
    data['file_entropy'] = ent
    size_kb = float(os.path.getsize(pe_path)) / 1024
    data['FileSize(KB)'] = size_kb


def extract_derived_features(pe_file, pe_path, data_dict):
    file_related_features(pe_file, data_dict)
    path_related_features(pe_path, data_dict)


def extract(file_path):
    try:
        pe_file = pefile.PE(file_path)
        empty_key_list = ["MD5", "e_cp", "e_cparhdr", "e_maxalloc", "e_sp", "e_lfanew", "MajorLinkerVersion",
                          "MinorLinkerVersion", "SizeOfCode", "SizeOfInitializedData", "SizeOfUninitializedData",
                          "AddressOfEntryPoint", "BaseOfCode", "MajorOperatingSystemVersion",
                          "MinorOperatingSystemVersion", "MajorImageVersion", "MinorImageVersion",
                          "MajorSubsystemVersion", "MinorSubsystemVersion", "CheckSum", "Subsystem",
                          "SizeOfStackReserve", "SizeOfStackCommit", "SizeOfHeapReserve", "SizeOfHeapCommit",
                          "LoaderFlags", "e_cblp_indicator", "BaseOfData", "has_warnings", "is_64bit",
                          "NumberOfSections", "article_indicator", "FH_char0", "FH_char1", "FH_char2", "FH_char3",
                          "FH_char4", "FH_char5", "FH_char6", "FH_char7", "FH_char8", "FH_char9", "FH_char10",
                          "FH_char11", "FH_char12", "FH_char13", "FH_char14", "OH_DLLchar0", "OH_DLLchar1",
                          "OH_DLLchar2", "OH_DLLchar3", "OH_DLLchar4", "OH_DLLchar5", "OH_DLLchar6", "OH_DLLchar7",
                          "OH_DLLchar8", "OH_DLLchar9", "OH_DLLchar10", "SusCreationYear", "FileInfo", "ImageBase",
                          "SectionAlignment", "FileAlignment", "SizeOfImage", "SizeOfHeaders", "ImportedLibsNum",
                          "ImportedFuncsNum", "DigitalSigned", "MinSectionEntropy", "MaxSectionEntropy",
                          "TotalSuspiciousSections", "TotalNonSuspiciousSections", "TotalSpecialSections",
                          ".text_entropy", ".data_entropy", "file_entropy", "FileSize(KB)"]
        empty_key_dict = {key: 0 for key in empty_key_list}
        empty_key_dict['MD5'] = get_md5(file_path)
        extract_raw_integrated_features(pe_file, empty_key_dict)
        extract_derived_features(pe_file, file_path, empty_key_dict)
        pe_file.close()
        return empty_key_dict
    except pefile.PEFormatError:
        return 0
