<div align="center">    
 
# MedAID-ML: A Multilingual Dataset of Biomedical Texts for Detecting AI-Generated Content

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![image](https://img.shields.io/pypi/pyversions/uv.svg)](https://pypi.python.org/pypi/uv)

</div>

## 📌&nbsp;&nbsp;Introduction
The original paper introduces MedAID-ML, which is a multilignual dataset of biomedical texts for detecting AI-generated content. The texts contain
(mainly) parallel data in English, German, Spanish and French, where AI-generated counterparts were created using three different state-of-the-art
Large Language Models (LLMs): mistral-7b, Llama3.1 and GPT-4o.

We conducted several baseline experiments as well as an XAI analysis. This repository contains the code for our experiments. For the dataset,
please refer to *Digital CSIC* or *TU Wien Research Data*.

Regarding questions, feel free to reach out to the authors of the paper!

## 📊&nbsp;&nbsp;Data Sources

### European Vaccination Information Portal (EVIP)

Also an agency of the European Union, the European Vaccination Information Portal (EVIP) provides accurate, objective, and up-to-date information on vaccines and vaccination. The portal offers disease factsheets containing key facts on various diseases, including symptoms, complications, risk factors, transmission methods, prevention, and treatment, as well as vaccination schedules, multimedia resources and also information for specific audiences.

The portal is available in all official EU languages, as well as Icelandic and Norwegian, ensuring broad accessibility across Europe. Some are automatically translated, which is fortunately not the case for our target languages.

[The factsheets](https://vaccination-info.europa.eu/en/disease-factsheets) are available in all languages, and consist of 20 texts each.

**Processing:** These were left as they were, as they have reasonable boundaries.

### Immunize

[Immunize.org](https://www.immunize.org/) (formerly known as the Immunization Action Coalition) is a U.S.-based organization dedicated to providing comprehensive immunization resources for healthcare professionals and the public.he website offers a wide array of materials, including: vaccine information statements detailing vaccine benefits and risks, clinical resources, educational materials and also vaccine news and updates.

Regarding language accessibility, Immunize.org provides resources in numerous languages. For instance, VISs are translated into over 40 languages, including Spanish, Chinese (Simplified and Traditional), Arabic, French, Russian, and Vietnamese. Additionally, clinical resources and patient handouts are available in multiple languages to ensure effective communication with non-English-speaking patients.

[VISs](https://www.immunize.org/vaccines/vis-translations) have been translated into several languages, but not all of them contain all VISs. They are given as PDFs, with 25 in Spanish, French and English, but only 21 in German. Only PDFs overlapping in all languages were used.

**Processing:** These were left as they were, as they have reasonable boundaries.

### Migration und Gesundheit - German Ministry of Health (BFG)

The "Migration und Gesundheit" portal, managed by the German Federal Ministry of Health, provides multilingual health information tailored for migrants and refugees. The website offers a wide array of resources, including information on the health care system, health and preventive care, long-term care and also addition and drugs.

The portal ensures accessibility by providing information in over 40 languages, including English, Turkish, Russian, Arabic, French, Spanish, and many others.

[Gesundheit für alle](https://www.migration-gesundheit.bund.de/fileadmin/Dateien/Publikationen/Gesundheit/wegweiser_gesundheit/deutsch.wegweiser-gesundheit.2022.pdf) is a PDF file that provides a guide to the German healthcare system, and it is available in Spanish, English and German.

**Processing:** Two topics, which were shorter than 100 words, were merged with the next one to ensure that context is preserved.

### European Food Safety Authority (EFSA)

The European Food Safety Authority (EFSA) provides a comprehensive range of data on its website, including food consumption and chemical/biological monitoring data as well as reports.

Regarding language accessibility, EFSA has expanded its online communications to all 24 official EU languages. Some of them have been translated automatically, none of them encompassing our targeted languages.

[The topics](https://www.efsa.europa.eu/en/topics) include a wide range of food-related information, including chemicals, materials, but also diseases. We chose only the onces we deem necessary for our goals, therefore including a total of 51 topics.

**Processing:** 8 articles had a wordcount of above 1350 (in Spanish, the 'longest' language), which is why we manually split it into parts of lengths lower than that. We manually ensured their correctness and alignment in all languages.

### Orphadata (INSERM)

The Orphanet organisation maintains a comprehensive knowledge base about rare diseases and orphan drugs, and releases it in re-usable and high-quality formats.

[Orphadata](https://www.orphadata.com/alignments/) is multilingual and is released in 12 official EU languages (Chinese, Czech, Dutch, English, French, German, Italian, Polish, Portuguese, Spanish, Turkish and Ukrainian). Files are available under the Commons Attribution 4.0 International (CC BY 4.0) licence.

We gathered definitions, signs and symptoms and phenotypes about 4389 rare diseases in English, German, Spanish and French.

**Processing:** Since each definition is roughly the same size and similar format, we simply group 5 definitions together to make the text per topic longer.

### Wikipedia

Wikipedia is a free, web-based, collaborative multilingual encyclopedia project supported by the non-profit Wikimedia Foundation. It contains a vast amount of information on a wide range of topics, including (bio)medical content, which is available in multiple languages. To ensure that the texts were not automatically generated, we only use articles that date back to before the release of ChatGPT, i.e. before 30th November 2022.

We extracted articles related to (bio)medical topics in English, German, Spanish and French.

**Processing:** First of all, some data cleaning was necessary, like removing excessive whitespaces, incorrect quotation marks etc. Furthermore, we removed all topics with less than 5 words. If topics had more than 9 sentences, we split them up into equally long parts, thereby ensuring that each file is only between 5 to 9 sentences long. From these split up files, we make sure that they contain a minimum of 100 words. (```processing.py```) Lastly, we take only those files that exist in all three languages. (```overlap.py```)

### Abstracts

We downloaded [PubMed](https://pubmed.ncbi.nlm.nih.gov/) abstracts available in English, Spanish, French and German.

### European Medicines Agency (EMA)

The [European Medicines Agency (EMA)](https://www.ema.europa.eu/) is an agency that supervises and evaluates pharmaceutical products of the European Union (EU).

We downloaded public assessment reports (EPARs) from 12 new medicinal products, which were published only from January 2025 to date. The goal is gathering data that might not have been used to train the LLMs in our experiments. 

Since EPARs are translated into all EU languages, we collected parallel data in English, Spanish, French and German.

### Cochrane

[Cochrane](https://www.cochrane.org/) is a database of meta-analyses and systematic reviews of updated results of clinical studies. We used abstracts of systematic reviews in all four languages. 

## 🔗&nbsp;&nbsp;Citation

```
TBA
```  
