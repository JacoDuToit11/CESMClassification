#----- Extracts information from different files, cleans data, and creates dataset -----#

# Imports
import pandas as pd
import math
from langdetect import detect

# Driver function
def main():
    preprocess()
    build_dataset()

# Combines information from the 3 files, cleans data and removes unwanted entries, fixes the dataset and saves to a file.
def preprocess():
    abstracts = pd.read_csv("data/abstracts.csv", names=['fake_id', 'id', 'abstract_content', 'language'])
    abstracts.drop(['fake_id'], axis = 1, inplace=True)

    # Remove all unwanted languages
    abstracts = abstracts[abstracts['language'] != ('a')]
    abstracts = abstracts[abstracts['language'] != ('a_ZA')]
    abstracts = abstracts[abstracts['language'] != ('aaf')]
    abstracts = abstracts[abstracts['language'] != ('ad_ZA')]
    abstracts = abstracts[abstracts['language'] != ('af')]
    abstracts = abstracts[abstracts['language'] != ('AF')]
    abstracts = abstracts[abstracts['language'] != ('af_ZA')]
    abstracts = abstracts[abstracts['language'] != ('AF-ZA')]
    abstracts = abstracts[abstracts['language'] != ('af-ZA')]
    abstracts = abstracts[abstracts['language'] != ('afr')]
    abstracts = abstracts[abstracts['language'] != ('afr_ZA')]
    abstracts = abstracts[abstracts['language'] != ('ar_ZA')]
    abstracts = abstracts[abstracts['language'] != ('de')]
    abstracts = abstracts[abstracts['language'] != ('dut')]
    abstracts = abstracts[abstracts['language'] != ('eaf_ZA')]
    abstracts = abstracts[abstracts['language'] != ('ef')]
    abstracts = abstracts[abstracts['language'] != ('ns')]
    abstracts = abstracts[abstracts['language'] != ('ns_ZA')]
    abstracts = abstracts[abstracts['language'] != ('nso')]
    abstracts = abstracts[abstracts['language'] != ('pe')]
    abstracts = abstracts[abstracts['language'] != ('pt_ZA')]
    abstracts = abstracts[abstracts['language'] != ('se')]
    abstracts = abstracts[abstracts['language'] != ('sf_ZA')]
    abstracts = abstracts[abstracts['language'] != ('sn')]
    abstracts = abstracts[abstracts['language'] != ('sn_ZA')]
    abstracts = abstracts[abstracts['language'] != ('so')]
    abstracts = abstracts[abstracts['language'] != ('sot')]
    abstracts = abstracts[abstracts['language'] != ('ss')]
    abstracts = abstracts[abstracts['language'] != ('ss_ZA')]
    abstracts = abstracts[abstracts['language'] != ('ss_ZA')]
    abstracts = abstracts[abstracts['language'] != ('st')]
    abstracts = abstracts[abstracts['language'] != ('st_ZA')]
    abstracts = abstracts[abstracts['language'] != ('tn')]
    abstracts = abstracts[abstracts['language'] != ('tn_ZA')]
    abstracts = abstracts[abstracts['language'] != ('ts')]
    abstracts = abstracts[abstracts['language'] != ('tsn')]
    abstracts = abstracts[abstracts['language'] != ('und')]
    abstracts = abstracts[abstracts['language'] != ('ve')]
    abstracts = abstracts[abstracts['language'] != ('ven')]
    abstracts = abstracts[abstracts['language'] != ('xh')]
    abstracts = abstracts[abstracts['language'] != ('xh_ZA')]
    abstracts = abstracts[abstracts['language'] != ('xho')]
    abstracts = abstracts[abstracts['language'] != ('xho_ZA')]
    abstracts = abstracts[abstracts['language'] != ('xo')]
    abstracts = abstracts[abstracts['language'] != ('zf_ZA')]
    abstracts = abstracts[abstracts['language'] != ('zu')]
    abstracts = abstracts[abstracts['language'] != ('zu_ZA')]
    abstracts = abstracts[abstracts['language'] != ('zul')]
    abstracts = abstracts[abstracts["abstract_content"].str.contains("AFRIKAANS:") == False]
    abstracts = abstracts[abstracts["abstract_content"].str.contains("Afrikaans:") == False]

    # Remove unwanted text at start of abstract
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH ABSTRACT: ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('English Abstract: ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH ABSTRACT : ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH ABSTRACT : ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH ABSTRACT :', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH ASTRACT: ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('English: ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH: ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('English : ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH : ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH SUMMARY: ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH SUMMARY : ', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH SUMMARY :', '')
    abstracts['abstract_content'] = abstracts.abstract_content.str.replace('ENGLISH SUMMURY :', '')

    # Remove all entries with abstracts that are less than 150 words.
    min_abstract_len = 150
    abstracts = abstracts[(abstracts['abstract_content'].str.count(' ').add(1) >= min_abstract_len)]

    documents = pd.read_csv("data/documents.csv", names=['id','link', 'title'])
    documents.drop(['title'], axis = 1, inplace=True)

    categories = pd.read_csv("data/categories.csv", encoding= 'unicode_escape')

    # Remove entries with no level 1 code
    categories = categories[categories.cesm_1_code.notnull()]
    categories = categories[categories.cesm_1_code != "unknown"]

    temp = pd.merge(documents, categories, on="link")

    data = pd.merge(temp, abstracts, on="id")
    data.drop_duplicates(subset ="id", keep = 'first', inplace = True)
    data.drop(['id', 'link', 'language'], axis = 1, inplace=True)

    for i in data.index:
        if (not math.isnan(data['cesm_3_code'][i])):
            data.at[i, 'cesm_3_code'] = str(data['cesm_3_code'][i])[:-2]
    
    ### Fix second level labels ###
    cesm_categories = pd.read_csv("data/cesm_categories.tsv", names=['code', 'description'], sep='\t')

    # Build dictionary of codes and descriptions
    cesm_map = dict()
    for i in range(0, len(cesm_categories)):
        if len(str(cesm_categories['code'][i])[:-2]) == 1:
            cesm_categories.at[i, 'code'] = '0' + str(cesm_categories['code'][i])
        elif len(str(cesm_categories['code'][i])[:-2]) == 3:
            cesm_categories.at[i, 'code'] = '0' + str(cesm_categories['code'][i])
        elif len(str(cesm_categories['code'][i])[:-2]) == 5:
            cesm_categories.at[i, 'code'] = '0' + str(cesm_categories['code'][i])
        cesm_map[cesm_categories['description'][i]] = str(cesm_categories['code'][i])[:-2]
    
    cesm_map['Social Work'] = '2008'
    cesm_map['Social Work, General'] = '200801'

    # Fix second level data by using third level data and the dictionary
    for i in data.index:
        if data['cesm_2_code'][i] == 'unknown':
            # not a valid cesm 2 code
            if (not math.isnan(float(data['cesm_3_code'][i]))):
                data.at[i, 'cesm_2_code'] = str(data['cesm_3_code'][i])[:-2]
            else:
                data.at[i, 'cesm_2_code'] = '0'
        elif (math.isnan(float(data['cesm_2_code'][i])) or len(data['cesm_2_code'][i]) < 3):
            # not a valid cesm 2 code
            if (not math.isnan(float(data['cesm_3_code'][i]))):
                data.at[i, 'cesm_2_code'] = str(data['cesm_3_code'][i])[:-2]
            else:
                data.at[i, 'cesm_2_code'] = '0'
        
        if len(data['cesm_2_code'][i]) < 3:
            if str(data['cesm_2_name'][i]) != 'nan':
                if (str(data['cesm_2_name'][i]).replace(u'\xa0', u' ') in cesm_map):
                    code_string = cesm_map[str(data['cesm_2_name'][i]).replace(u'\xa0', u' ')]
                    if (len(code_string) == 3 or len(code_string) == 4):
                        data.at[i, 'cesm_2_code'] = cesm_map[str(data['cesm_2_name'][i]).replace(u'\xa0', u' ')]
        elif len(data['cesm_2_code'][i]) > 4:
            data.at[i, 'cesm_2_code'] = str(data['cesm_2_code'][i])[:-2]
        
        if math.isnan(float(data['cesm_3_code'][i])):
            data.at[i, 'cesm_3_code'] = '0'

        if (len(data['cesm_1_code'][i]) <= 1):
                data.at[i, 'cesm_1_code'] = '0' + data['cesm_1_code'][i]

        if data['cesm_2_code'][i] != '0':
            if (len(data['cesm_2_code'][i]) <= 3):
                data.at[i, 'cesm_2_code'] = '0' + data['cesm_2_code'][i]

        if data['cesm_3_code'][i] != '0':
            if (len(data['cesm_3_code'][i]) <= 5):
                data.at[i, 'cesm_3_code'] = '0' + data['cesm_3_code'][i]

    # Combine title and abstract and remove entries titles if they are not english.
    check_language = True
    if check_language:
        for i in data.index:
            if detect(data['title'][i]) != 'en':
                # do not use title if not english
                data.at[i, 'content'] = data['abstract_content'][i]
            else:
                data.at[i, 'content'] = data['title'][i] + ". " + data['abstract_content'][i]
    else:
        data['content'] = data['title'] + ". " + data['abstract_content']

    data.drop(labels = ['title', 'abstract_content'], axis = 1, inplace = True)

    ### Fix dataset ###
    data.reset_index(inplace = True)
    for i in data.index:
        # Check that all three levels are consistent
        if data['cesm_2_code'][i] != '0':
            if data['cesm_1_code'][i] != data['cesm_2_code'][i][0:2]:
                data.at[i, 'cesm_1_code'] = '0'
            elif data['cesm_3_code'][i] != '0' and data['cesm_2_code'][i] != data['cesm_3_code'][i][0:4]:
                # Third levels are correct sometimes when second level is wrong, so we change those entries
                data.at[i, 'cesm_2_code'] = data['cesm_3_code'][i][0:4]
        if data['cesm_1_code'][i] == '07':
            if str(data['cesm_2_name'][i])[0:37] == 'Educational Management and Leadership':
                data.at[i, 'cesm_2_code'] = '0703'
        
        if not data['cesm_3_code'][i] == '0':
            if not data['cesm_3_name'][i] in cesm_map or not cesm_map[str(data['cesm_3_name'][i])] == data['cesm_3_code'][i]:
                data.at[i, 'cesm_1_code'] = '0'
            else:
                data.at[i, 'cesm_1_code'] = data['cesm_3_code'][i][0:2]
        
        if data['cesm_1_code'][i] == '12' and data['cesm_1_name'][i] == 'Languages, Linguistics and Literature':
            data.at[i, 'cesm_1_code'] = '11'
            data.at[i, 'cesm_2_code'] = '0'
            data.at[i, 'cesm_3_code'] = '0'
        
        if data['cesm_1_code'][i] == '13':
            if data['cesm_1_name'][i] == 'Physical Education, Health Education and Leisure':
                data.at[i, 'cesm_1_code'] = '0'
            elif data['cesm_1_name'][i] == 'Law':
                data.at[i, 'cesm_1_code'] = '12'
                data.at[i, 'cesm_2_code'] = '0'
                data.at[i, 'cesm_3_code'] = '0'
            
        if (data['cesm_1_code'][i] == '14') and (data['cesm_1_name'][i] == 'Libaries and Museums' or data['cesm_1_name'][i] == 'Libraries and Museums' or data['cesm_1_name'][i] == 'Social Sciences and Social Studies'):
            data.at[i, 'cesm_1_code'] = '0'
        
        if (data['cesm_1_code'][i] == '15'):
            if (data['cesm_1_name'][i] == 'Life Sciences'):
                data.at[i, 'cesm_1_code'] = '13'
            elif data['cesm_1_name'][i] == 'Life Sciences and Physical Sciences':
                data.at[i, 'cesm_1_code'] = '0'
        
        if (data['cesm_1_code'][i] == '15'):
            if (data['cesm_1_name'][i] == 'Life Sciences'):
                data.at[i, 'cesm_1_code'] = '13'
            elif data['cesm_1_name'][i] == 'Life Sciences and Physical Sciences':
                data.at[i, 'cesm_1_code'] = '0'
        
        if (data['cesm_1_code'][i] == '16'):
            if (data['cesm_1_name'][i] == 'Mathematical Sciences'):
                data.at[i, 'cesm_1_code'] = '15'
        
        if (data['cesm_1_code'][i] == '18'):
            if (data['cesm_1_name'][i] == 'Philosphy, Religion and Theology' or data['cesm_1_name'][i] == 'Philosophy, Religion and Theology'):
                data.at[i, 'cesm_1_code'] = '17'
        
        if (data['cesm_1_code'][i] == '19'):
            if (data['cesm_1_name'][i] == 'Physical Education, Health Education and Leisure'):
                data.at[i, 'cesm_1_code'] = '0'
        
        if (data['cesm_1_code'][i] == '20'):
            if (data['cesm_1_name'][i] == 'Psychology'):
                data.at[i, 'cesm_1_code'] = '18'
                data.at[i, 'cesm_2_code'] = '0'
                data.at[i, 'cesm_3_code'] = '0'
        
        if (data['cesm_1_code'][i] == '21'):
            data.at[i, 'cesm_1_code'] = '19'
            data.at[i, 'cesm_2_code'] = '0'
            data.at[i, 'cesm_3_code'] = '0'
        
        if (data['cesm_1_code'][i] == '22'):
            if (data['cesm_2_name'][i] == 'Economics'):
                data.at[i, 'cesm_1_code'] = '04'
                data.at[i, 'cesm_2_code'] = '0404'
                data.at[i, 'cesm_3_code'] = '0'
            else:
                data.at[i, 'cesm_1_code'] = '20'
                data.at[i, 'cesm_2_code'] = '0'
                data.at[i, 'cesm_3_code'] = '0'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Education'):
            if (data['cesm_3_name'][i] == 'Curriculum Studies'):
                data.at[i, 'cesm_1_code'] = '07'
                data.at[i, 'cesm_2_code'] = '0702'
                data.at[i, 'cesm_3_code'] = '070201'
            elif (data['cesm_3_name'][i] == 'Education Studies'):
                data.at[i, 'cesm_1_code'] = '07'
                data.at[i, 'cesm_2_code'] = '0701'
                data.at[i, 'cesm_3_code'] = '070101'
            elif (data['cesm_3_name'][i] == 'Educational Leadership and Management, General'):
                data.at[i, 'cesm_1_code'] = '07'
                data.at[i, 'cesm_2_code'] = '0703'
                data.at[i, 'cesm_3_code'] = '070301'
            elif (data['cesm_3_name'][i] == 'Mathematics  ECD and GET'):
                data.at[i, 'cesm_1_code'] = '07'
                data.at[i, 'cesm_2_code'] = '0711'
                data.at[i, 'cesm_3_code'] = '071112'
            elif (data['cesm_3_name'][i] == 'Natural Sciences  ECD and GET'):
                data.at[i, 'cesm_1_code'] = '07'
                data.at[i, 'cesm_2_code'] = '0711'
                data.at[i, 'cesm_3_code'] = '071113'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Engineering and Engineering Technology'):
            if (data['cesm_3_name'][i] == 'Biomedical Systems'):
                data.at[i, 'cesm_1_code'] = '08'
                data.at[i, 'cesm_2_code'] = '0804'
                data.at[i, 'cesm_3_code'] = '080401'
            
        if (data['cesm_1_code'][i] == '0' and (data['cesm_1_name'][i] == 'Health Care and Health Sciences' or data['cesm_1_name'][i] == 'Health Professions and Related Clinical Sciences')):
            if (data['cesm_3_name'][i] == 'Haematology'):
                data.at[i, 'cesm_1_code'] = '09'
                data.at[i, 'cesm_2_code'] = '0907'
                data.at[i, 'cesm_3_code'] = '090727'
            elif (data['cesm_3_name'][i] == 'Immunology'):
                data.at[i, 'cesm_1_code'] = '09'
                data.at[i, 'cesm_2_code'] = '0907'
                data.at[i, 'cesm_3_code'] = '090702'
            elif (data['cesm_3_name'][i] == 'Pathology'):
                data.at[i, 'cesm_1_code'] = '09'
                data.at[i, 'cesm_2_code'] = '0907'
                data.at[i, 'cesm_3_code'] = '090747'
            elif (data['cesm_3_name'][i] == 'Nursing'):
                data.at[i, 'cesm_1_code'] = '09'
                data.at[i, 'cesm_2_code'] = '0908'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_3_name'][i] == 'Pharmacology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1309'
                data.at[i, 'cesm_3_code'] = '130901'
            elif (data['cesm_3_name'][i] == 'Psychiatry'):
                data.at[i, 'cesm_1_code'] = '09'
                data.at[i, 'cesm_2_code'] = '0907'
                data.at[i, 'cesm_3_code'] = '090709'
            elif (data['cesm_3_name'][i] == 'Virology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1305'
                data.at[i, 'cesm_3_code'] = '130503'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Languages, Linguistics and Literature'):
            if (data['cesm_2_name'][i] == 'English Language and Literature'):
                data.at[i, 'cesm_1_code'] = '11'
                data.at[i, 'cesm_2_code'] = '1102'
                data.at[i, 'cesm_3_code'] = '110201'
            elif (data['cesm_2_name'][i] == 'European Languages and Literature (excluding Dutch)'):
                data.at[i, 'cesm_1_code'] = '11'
                data.at[i, 'cesm_2_code'] = '1115'
                data.at[i, 'cesm_3_code'] = '111501'
            elif (data['cesm_2_name'][i] == 'Middle/Near Eastern and Semitic Languages and Literature'):
                data.at[i, 'cesm_1_code'] = '11'
                data.at[i, 'cesm_2_code'] = '1117'
                data.at[i, 'cesm_3_code'] = '0'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Life Sciences'):
            if (data['cesm_2_name'][i] == 'Life Sciences and Physical Sciences'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '0'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_3_name'][i] == 'Genetics'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1307'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_3_name'][i] == 'Molecular Biology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1302'
                data.at[i, 'cesm_3_code'] = '130203'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Life Sciences and Physical Sciences'):
            if (data['cesm_2_name'][i] == 'Zoology/Animal Biology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1306'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_2_name'][i] == 'Chemistry'):
                data.at[i, 'cesm_1_code'] = '14'
                data.at[i, 'cesm_2_code'] = '1404'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_2_name'][i] == 'Geology and Earth Sciences/Geosciences'):
                data.at[i, 'cesm_1_code'] = '14'
                data.at[i, 'cesm_2_code'] = '1406'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_2_name'][i] == 'Physics'):
                data.at[i, 'cesm_1_code'] = '14'
                data.at[i, 'cesm_2_code'] = '1407'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_3_name'][i] == 'Genetics'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1307'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_3_name'][i] == 'Microbiology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1305'
                data.at[i, 'cesm_3_code'] = '130501'
            elif (data['cesm_3_name'][i] == 'Molecular Biology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1302'
                data.at[i, 'cesm_3_code'] = '130203'
            elif (data['cesm_3_name'][i] == 'Plant Pathology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1303'
                data.at[i, 'cesm_3_code'] = '130302'
            elif (data['cesm_3_name'][i] == 'General Zoology'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1306'
                data.at[i, 'cesm_3_code'] = '130601'
            elif (data['cesm_3_name'][i] == 'Biochemistry'):
                data.at[i, 'cesm_1_code'] = '13'
                data.at[i, 'cesm_2_code'] = '1302'
                data.at[i, 'cesm_3_code'] = '130201'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Mathematical Sciences'):
            if (data['cesm_2_name'][i] == 'Mathematics'):
                data.at[i, 'cesm_1_code'] = '15'
                data.at[i, 'cesm_2_code'] = '1501'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_2_name'][i] == 'Applied Mathematics'):
                data.at[i, 'cesm_1_code'] = '15'
                data.at[i, 'cesm_2_code'] = '1502'
                data.at[i, 'cesm_3_code'] = '0'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Philosophy, Religion and Theology'):
            if (data['cesm_2_name'][i] == 'Philosophy'):
                data.at[i, 'cesm_1_code'] = '17'
                data.at[i, 'cesm_2_code'] = '1701'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_3_name'][i] == 'Old and New Testament Science' or data['cesm_3_name'][i] == 'Christianity' or data['cesm_3_name'][i] == 'Missiology'):
                data.at[i, 'cesm_1_code'] = '17'
                data.at[i, 'cesm_2_code'] = '1703'
                data.at[i, 'cesm_3_code'] = '170303'
            elif (data['cesm_2_name'][i] == 'Theology'):
                data.at[i, 'cesm_1_code'] = '17'
                data.at[i, 'cesm_2_code'] = '1703'
                data.at[i, 'cesm_3_code'] = '0'

        if (data['cesm_1_code'][i] == '0' and (data['cesm_1_name'][i] == 'Physical Sciences' or data['cesm_1_name'][i] == 'Life Sciences and Physical Sciences')):
            if (data['cesm_3_name'][i] == 'Polymers'):
                data.at[i, 'cesm_1_code'] = '14'
                data.at[i, 'cesm_2_code'] = '1404'
                data.at[i, 'cesm_3_code'] = '140406'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Psychology'):
            if (data['cesm_2_name'][i] == 'Educational Psychology' or data['cesm_2_name'][i] == 'Psychology applied to Education'):
                data.at[i, 'cesm_1_code'] = '18'
                data.at[i, 'cesm_2_code'] = '1808'
                data.at[i, 'cesm_3_code'] = '0'
            elif (data['cesm_2_name'][i] == 'Industrial and Organisational Psychology' or data['cesm_3_name'][i] == 'Industrial Psychology'):
                data.at[i, 'cesm_1_code'] = '18'
                data.at[i, 'cesm_2_code'] = '1814'
                data.at[i, 'cesm_3_code'] = '0'
            
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Public Administration and Social Services'):
            data.at[i, 'cesm_1_code'] = '20'
            data.at[i, 'cesm_2_code'] = '2007'
            data.at[i, 'cesm_3_code'] = '200701'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Public Management and Services'):
            if (data['cesm_3_name'][i] == 'Criminalistics and Criminal Science'):
                data.at[i, 'cesm_1_code'] = '19'
                data.at[i, 'cesm_2_code'] = '1905'
                data.at[i, 'cesm_3_code'] = '190507'
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Social Sciences'):
            if (data['cesm_2_name'][i] in cesm_map):
                data.at[i, 'cesm_1_code'] = '20'
            else:                
                data.at[i, 'cesm_1_code'] = '20'
                data.at[i, 'cesm_2_code'] = '0'
                data.at[i, 'cesm_3_code'] = '0' 
        
        if (data['cesm_1_code'][i] == '0' and data['cesm_1_name'][i] == 'Social Sciences and Social Studies'):
            if (data['cesm_2_name'][i] == 'Anthropology' or data['cesm_2_name'][i] == 'History'):
                data.at[i, 'cesm_1_code'] = '20'
            elif (data['cesm_3_name'][i] == 'Geography' or data['cesm_3_name'][i] == 'Environmental Science'):
                data.at[i, 'cesm_1_code'] = '14'
            elif (data['cesm_3_name'][i] == 'Social-Cultural Anthropology'):
                data.at[i, 'cesm_1_code'] = '20'
        
        if (data['cesm_1_code'][i] == '20' and data['cesm_2_code'][i] == '2009'):
            data.at[i, 'cesm_2_code'] = '0'

    # Set non-english entries to invalid.
    data.reset_index(inplace = True)
    if check_language:
        for i in data.index:
            if detect(data['content'][i]) != 'en':
                data.at[i, 'cesm_1_code'] = '0'

    # Remove all invalid entries
    data = data[data['cesm_1_code'] != '0']
    data.sort_values(by=['cesm_1_code','cesm_2_code', 'cesm_3_code'], inplace=True)

    data.drop(labels = ['cesm_1_name','cesm_2_name', 'cesm_3_name'], axis = 1, inplace = True)
    data.to_csv('data/normal_data.csv', index = False)

# Builds the dataset used for all further tasks, determines which classes to include or remove.
def build_dataset():
    data = pd.read_csv('data/normal_data.csv')
    data.drop(labels = ['level_0', 'index'], axis = 1, inplace = True)

    # Obtain and sort level 1 labels
    level_1_labels = data.cesm_1_code.unique()
    level_1_labels = list(level_1_labels)
    level_1_labels.sort(key=float)

    # Obtain and sort level 2 labels
    level_2_labels = data.cesm_2_code.unique()
    level_2_labels = list(level_2_labels)
    level_2_labels.sort(key=float)
    level_2_labels = level_2_labels[1:]

    all_labels = [*level_1_labels, *level_2_labels]

    cesm_1_codes = pd.get_dummies(data.cesm_1_code)
    num_level_1_codes = len(cesm_1_codes.columns)
    cesm_2_codes = pd.get_dummies(data.cesm_2_code)
    data = data.join(cesm_1_codes)
    data = data.join(cesm_2_codes)

    data.drop([0], axis = 1, inplace = True)

    # Check which first level labels have enough entries
    level_1_labels = data.columns.values[4:4+num_level_1_codes]
    min_number_level1 = 550
    remove_labels = []
    for label in level_1_labels:
        count = data[label].sum()
        if count < min_number_level1:
            remove_labels.append(label)

    number_labels_remove = len(remove_labels)
    data.query('cesm_1_code not in @remove_labels', inplace = True)
    
    # # Removing level 2 labels of categories that have been removed
    level_2_labels = data.columns.values[4+num_level_1_codes:]
    for i in range(0, len(level_2_labels)):
        for j in range(0, number_labels_remove):
            if len(str(remove_labels[j])) == 1:
                if len(str(level_2_labels[i])) == 3:
                    if int(str(level_2_labels[i])[0]) == remove_labels[j]:
                        remove_labels.append(level_2_labels[i])
            elif len(str(remove_labels[j])) == 2:
                if len(str(level_2_labels[i])) == 4:
                    if int(str(level_2_labels[i])[0:2]) == remove_labels[j]:
                        remove_labels.append(level_2_labels[i])

    data.drop(remove_labels, axis = 1, inplace = True)
    level_1_labels = data.columns.values[4:4+11]

    ### Exploratory data analysis to decide which labels to discard ###
    counts = []
    for label in level_1_labels:
        counts.append((label, data[label].sum()))
    level_1_stats = pd.DataFrame(counts, columns=['label', 'number_theses'])
    print("Level 1 stats:")
    print(level_1_stats)

    all_labels = data.columns.values[4:]
    print("All labels:")
    print(all_labels)

    level_2_start_labels = [101, 401, 701, 802, 901, 1101, 1301, 1402, 1701, 1802, 2001]
    level_2_start_indices = []
    level_2_end_index = 0
    for i in range(0, len(all_labels)):
        if all_labels[i] in level_2_start_labels:
            level_2_start_indices.append(i)
        elif all_labels[i] == 2008:
            level_2_end_index = i
    
    level_2_stats = []
    for i in range(1, len(level_2_start_indices)):
        temp_labels = all_labels[level_2_start_indices[i-1]:level_2_start_indices[i]]
        counts = []
        for label in temp_labels:
            counts.append((label, data[label].sum()))
        level_2_stats.append(pd.DataFrame(counts, columns=['label', 'number_theses']))

    temp_labels = all_labels[level_2_start_indices[len(level_2_start_indices)-1]:level_2_end_index]
    counts = []
    for label in temp_labels:
        counts.append((label, data[label].sum()))
    level_2_stats.append(pd.DataFrame(counts, columns=['label', 'number_theses']))

    print("Level 2 stats: ")
    for i in range(0, len(level_2_stats)):
        print(level_2_stats[i])
        print()

    # Remove labels from level 2 that do not have enough entries
    level_2_thresholds = [89, 200, 170, 110, 160, 95, 145, 130, 73, 70, 130]

    remove_labels = []
    for i in range(0, len(level_2_stats)):
        for label in level_2_stats[i]['label']:
            count = data[label].sum()
            if count < level_2_thresholds[i]:
                remove_labels.append(label)
    
    def other_class(level_1_class, remove_labels, row):
        has_other_class = 0
        for label in remove_labels:
            if level_1_class < 10 and len(str(label)) == 3:
                if int(str(label)[0]) == level_1_class:
                    if row[label] == 1:
                        has_other_class = 1
                        break
            elif level_1_class >= 10 and len(str(label)) == 4:
                if int(str(label)[0:2]) == level_1_class:
                    if row[label] == 1:
                        has_other_class = 1
                        break
        return has_other_class

    data[199] = data.apply(lambda row: other_class(1, remove_labels, row), axis = 1) 
    data[499] = data.apply(lambda row: other_class(4, remove_labels, row), axis = 1) 
    data[799] = data.apply(lambda row: other_class(7, remove_labels, row), axis = 1) 
    data[899] = data.apply(lambda row: other_class(8, remove_labels, row), axis = 1) 
    data[999] = data.apply(lambda row: other_class(9, remove_labels, row), axis = 1) 
    data[1199] = data.apply(lambda row: other_class(11, remove_labels, row), axis = 1) 
    data[1399] = data.apply(lambda row: other_class(13, remove_labels, row), axis = 1) 
    data[1499] = data.apply(lambda row: other_class(14, remove_labels, row), axis = 1) 
    data[1799] = data.apply(lambda row: other_class(17, remove_labels, row), axis = 1) 
    data[1899] = data.apply(lambda row: other_class(18, remove_labels, row), axis = 1) 
    data[2099] = data.apply(lambda row: other_class(20, remove_labels, row), axis = 1) 
    
    data.drop(remove_labels, axis = 1, inplace = True)

    # Remove unnessary columns and save to file
    data.drop(labels = ['cesm_1_code', 'cesm_2_code', 'cesm_3_code'], axis = 1, inplace = True)
    sorted_cols = sorted(data.columns.values[1:])
    sorted_cols = ['content'] + sorted_cols
    data = data[sorted_cols]
    data.to_csv('data/data.csv', index = False)

# Obtains the dataset
def get_dataset():
    data = pd.read_csv('data/data.csv')
    content = data['content'].values
    return content, data.loc[:, '1':]

# Obtains the dataset after cleaning (for Tokenizer)
def get_dataset_clean():
    data = pd.read_csv('data/data.csv')
    cleanData(data)
    content = data['content'].values
    return content, data.loc[:, '1':]

# Cleans data
def cleanData(data):
    # Preprocessing
    def decontract(sentence):
        sentence = re.sub(r"n\'t", " not", sentence)
        sentence = re.sub(r"\'re", " are", sentence)
        sentence = re.sub(r"\'s", " is", sentence)
        sentence = re.sub(r"\'d", " would", sentence)
        sentence = re.sub(r"\'ll", " will", sentence)
        sentence = re.sub(r"\'t", " not", sentence)
        sentence = re.sub(r"\'ve", " have", sentence)
        sentence = re.sub(r"\'m", " am", sentence)
        return sentence
    
    def removePunctuation(sentence): 
        sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)
        sentence = sentence.strip()
        sentence = sentence.replace("\n"," ")
        return sentence
    
    def removeStopWords(sentence):
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
        return pattern.sub('', sentence)
    
    def stemming(sentence):
        stemmer = SnowballStemmer("english")
        stemmedSentence = ""
        for word in sentence.split():
            stem = stemmer.stem(word)
            stemmedSentence += stem
            stemmedSentence += " "
        stemmedSentence = stemmedSentence.strip()
        return stemmedSentence

    data['content'] = data['content'].map(lambda x: x.lower())
    for i in range(0, len(data)):
        data['content'][i] = decontract(data['content'][i])
        data['content'][i] = removePunctuation(data['content'][i])
        data['content'][i] = removeStopWords(data['content'][i])
        data['content'][i] = stemming(data['content'][i])

if __name__ == '__main__':
    main()