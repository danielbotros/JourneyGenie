from re import S
import requests
import csv
from bs4 import BeautifulSoup
import sys
import re


def processDogBreedInfo():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:55.0) Gecko/20100101 Firefox/55.0', }

    dog_breed_info = []
  # INITIALIZE CSV
    file = open('export_data.csv', 'w', newline='')
    writer = csv.writer(file)
    csvheaders = [' Name', 'Description', 'Temperament',
                  'Image', 'Price', 'Hypoallergenic']
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

        # scrapes the urls of the images
        image_url = "https://www.dogbreedslist.info" + \
            breed_data_soup.find(
                'div', attrs={'class': 'Slides fade'}).find('img')['src']
        dog_breed['image'] = image_url

        # attempt to download images to the backend/images/ folder
        # image_data = requests.get(image_url).content
        # image_file_name = dog_breed['name'].replace(" ", "") + '.jpg'
        # image_file_path = 'images/' + image_file_name
        # with open(image_file_path, "wb") as f:
        #     f.write(image_data)
        # dog_breed['image'] = image_file_name

        # gets the average prices
        price = breed_data_soup.find(
            'p', attrs={'class': 'price'}).find_previous_sibling().text
        dog_breed['price'] = price

        # gets whether or not a dog breed is hypoallergenic
        try:
            hypoallergenic = breed_data_soup.find('table', attrs={'class': 'table-02'}).find(
                'p', string=re.compile('Hypoallergenic')).get_text(strip=False)
        except:
            try:
                hypoallergenic = breed_data_soup.find('table', attrs={'class': 'table-02'}).find('td', string=re.compile(
                    'Health Issues')).find_next_sibling().find('p').find_next_sibling().find('span').get_text(strip=False)
            except:
                try:
                    hypoallergenic_index = breed_data_soup.find('table', attrs={'class': 'table-02'}).find(
                        'td', string=re.compile('Health Issues')).find_next_sibling().text.index('Hypoallergenic')
                    hypoallergenic = breed_data_soup.find('table', attrs={'class': 'table-02'}).find(
                        'td', string=re.compile('Health Issues')).find_next_sibling().text[hypoallergenic_index:]
                except:
                    hypoallergenic = "Hypoallergenic: Unknown"
        dog_breed['hypoallergenic'] = hypoallergenic

        file = open('export_data.csv', 'a', newline='', encoding='utf-8')
        writer = csv.writer(file)
        newheaders = ([finalname, description, temperament,
                      image_url, price, hypoallergenic])
        writer.writerow(newheaders)
        dog_breed_info.append(dog_breed)
    return dog_breed_info


processDogBreedInfo()
