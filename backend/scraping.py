from re import S
import requests
import csv
from bs4 import BeautifulSoup
import sys


def processDogBreedInfo():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0', }

    dog_breed_info = []
  # INITIALIZE CSV
    file = open('export_data.csv', 'w', newline='')
    writer = csv.writer(file)
    csvheaders = [' Name', 'Description', 'Temperament']
    writer.writerow(csvheaders)
    # add each individual item to csv
    file = open('export_data.csv', 'a', newline='', encoding='utf-8')
    # Scrape the dog names
    URLname = "https://www.dogbreedslist.info/dog-breeds-a-z/"
    rnames = requests.get(URLname, headers=headers)
    soupnames = BeautifulSoup(rnames.content, 'html5lib')
    namecol = soupnames.find('div', attrs={'class': 'all-a-z'})
    dog_names = namecol.find_all("dd")
    for dog in dog_names:
        dog_breed = {}
        name = dog.find("a")
        emtag = name.find("em")
        emtag.extract()
        span = name.find("span")
        span.extract()
        finalname = name.text
        dog_breed['name'] = finalname
        a_href = name.get("href")
        dog_breed['link'] = a_href
        r_dogdata = requests.get(
            "https://www.dogbreedslist.info"+a_href, headers=headers)
        breed_data_soup = BeautifulSoup(r_dogdata.content, 'html5lib')
        try:
            description = breed_data_soup.find(
                'table', attrs={'class': 'table-04'}).find('div', attrs={'class': 'fold-text'}).select('p')[0].get_text(strip=False)
            temperament = breed_data_soup.find(
                'table', attrs={'class': 'table-04'}).find('div', attrs={'class': 'fold-text'}).select('p')[1].get_text(strip=False)
        except:
            try:
                description = breed_data_soup.find(
                    'table', attrs={'class': 'table-04'}).select('tbody > tr')[1].select('td > p')[0].get_text(strip=False)
                temperament = breed_data_soup.find(
                    'table', attrs={'class': 'table-04'}).select('tbody > tr')[1].select('td > p')[1].get_text(strip=False)
            except:
                description = ""
                temperament = breed_data_soup.find(
                    'table', attrs={'class': 'table-04'}).select('tbody > tr')[1].select('td > p')[0].get_text(strip=False)

        dog_breed['description'] = description
        dog_breed['temperament'] = temperament
        file = open('export_data.csv', 'a', newline='', encoding='utf-8')
        writer = csv.writer(file)
        newheaders = ([finalname, description, temperament])
        writer.writerow(newheaders)
        dog_breed_info.append(dog_breed)
    return dog_breed_info


processDogBreedInfo()
